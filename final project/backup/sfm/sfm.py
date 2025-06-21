
@njit
def fast_reprojection_error(xyz_3d, R, t, K, observed_pt):
    xyz_cam = R @ xyz_3d + t
    if xyz_cam[2] <= 0:
        return np.inf
    uv = K @ xyz_cam
    if abs(uv[2]) < 1e-8:
        return np.inf
    proj = uv[:2] / uv[2]
    return np.linalg.norm(proj - observed_pt)

@njit
def fast_triangulation(P1, P2, pt1, pt2):
    A = np.zeros((4,4))
    A[0] = pt1[0] * P1[2] - P1[0]
    A[1] = pt1[1] * P1[2] - P1[1]
    A[2] = pt2[0] * P2[2] - P2[0]
    A[3] = pt2[1] * P2[2] - P2[1]
    U, s, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3] if abs(X[3])>1e-8 else np.zeros(3)

@njit
def compute_triangulation_angle(pt3d, c1, c2):
    ray1 = pt3d - c1
    ray2 = pt3d - c2
    n1 = np.linalg.norm(ray1); n2 = np.linalg.norm(ray2)
    if n1==0 or n2==0:
        return 0.0
    cosang = np.dot(ray1, ray2)/(n1*n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

class FastIncrementalSfM:
    def __init__(self, K_init: np.ndarray, optimize_intrinsics: bool=True, num_threads: int=8):
        self.K_init = K_init.copy()
        self.optimize_intrinsics = optimize_intrinsics
        self.num_threads = num_threads
        self.cameras: Dict[int, Camera] = {}
        self.points_3d: Dict[int, Point3D] = {}
        self.features: Dict[int, Tuple[List, np.ndarray]] = {}
        self.matches: Dict[Tuple[int,int], List] = {}
        self.feature_tree: Dict[int, NearestNeighbors] = {}
        self.covisibility_graph = defaultdict(set)
        self.descriptor_cache: Dict[int, np.ndarray] = {}
        self.pose_cache: Dict[int, Tuple[np.ndarray,np.ndarray]] = {}
        self.pnp_threshold = 4.0
        self.triangulation_threshold = 2.0
        self.min_triangulation_angle = 5.0
        self.max_reprojection_error = 4.0
        self.min_track_length = 2
        self.batch_size = 1000
        self.early_termination_ratio = 0.8
        self.adaptive_threshold_factor = 0.9

    def preprocess_features(self):
        def build(idx):
            if idx not in self.features: return idx, None
            kp, desc = self.features[idx]
            if desc is None or len(desc)==0: return idx, None
            tree = NearestNeighbors(n_neighbors=50, algorithm='kd_tree', metric='cosine')
            tree.fit(desc)
            return idx, tree
        with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
            for idx, tree in ex.map(build, self.features.keys()):
                if tree: self.feature_tree[idx]=tree
        self.build_covisibility_graph()

    def build_covisibility_graph(self):
        for (i,j), m in self.matches.items():
            if len(m)>20:
                self.covisibility_graph[i].add(j)
                self.covisibility_graph[j].add(i)

    def fast_guided_matching(self, img_idx: int, radius: float=30.0):
        if img_idx in self.pose_cache:
            R0,t0 = self.pose_cache[img_idx]
        else:
            p = self.estimate_initial_pose(img_idx)
            if p is None: return []
            R0,t0 = p; self.pose_cache[img_idx]=p
        if not self.points_3d: return []
        pts = np.array([p.xyz for p in self.points_3d.values()])
        cams = list(self.points_3d.keys())
        xyz_cam = (R0@pts.T).T + t0
        mask = xyz_cam[:,2]>0
        uv = (self.K_init@(xyz_cam[mask].T)).T
        z = uv[:,2]; valid = mask.copy(); valid[mask]=np.abs(z)>1e-8
        pts2d = uv[valid, :2]/z[valid,None]
        kp,desc = self.features[img_idx]
        tree = self.feature_tree.get(img_idx,None)
        out=[]
        for idx,proj in zip(np.where(valid)[0], pts2d):
            pid=list(self.points_3d.keys())[idx]
            if pid not in self.descriptor_cache: continue
            d0 = self.descriptor_cache[pid]
            dists,inds = tree.radius_neighbors([proj], radius)
            best=None;sim=0
            for i in inds[0]:
                d1=desc[i]; s=np.dot(d0,d1)/(np.linalg.norm(d0)*np.linalg.norm(d1)+1e-8)
                if s>sim and s>0.8: sim, best=s,i
            if best is not None: out.append((pid,best))
        return out

    def parallel_triangulation(self, img_idx: int):
        tasks=[]
        for ref in self.cameras:
            if ref==img_idx: continue
            key=(min(ref,img_idx),max(ref,img_idx))
            if key not in self.matches or len(self.matches[key])<10: continue
            tasks.append((ref,img_idx,key))
        def work(task):
            r,i,k=task; kp1,_=self.features[r];kp2,_=self.features[i]
            cam1,cam2=self.cameras[r],self.cameras[i]
            P1=cam1.K@np.hstack([cam1.R,cam1.t.reshape(-1,1)])
            P2=cam2.K@np.hstack([cam2.R,cam2.t.reshape(-1,1)])
            pts1=np.array([kp1[m.queryIdx].pt for m in self.matches[k]])
            pts2=np.array([kp2[m.trainIdx].pt for m in self.matches[k]])
            pts4d=cv2.triangulatePoints(P1,P2,pts1.T,pts2.T)
            pts3d=(pts4d[:3]/pts4d[3]).T
            new={}
            c1=-cam1.R.T@cam1.t; c2=-cam2.R.T@cam2.t
            for idx,(m,pt) in enumerate(zip(self.matches[k],pts3d)):
                if compute_triangulation_angle(pt,c1,c2)<self.min_triangulation_angle: continue
                e1=fast_reprojection_error(pt,cam1.R,cam1.t,cam1.K,np.array(kp1[m.queryIdx].pt))
                e2=fast_reprojection_error(pt,cam2.R,cam2.t,cam2.K,np.array(kp2[m.trainIdx].pt))
                if e1>self.max_reprojection_error or e2>self.max_reprojection_error: continue
                pid=len(self.points_3d)+len(new)
                new[pid]=Point3D(xyz=pt, color=np.array([128,128,128]),
                                 error=max(e1,e2),
                                 observations={r:m.queryIdx,i:m.trainIdx},
                                 descriptor=self.features[r][1][m.queryIdx])
            return new
        with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
            for res in ex.map(work,tasks): self.points_3d.update(res)

    def smart_candidate_selection(self):
        scores={}
        for idx in range(len(self.features)):
            if idx in self.cameras: continue
            score=len(self.fast_guided_matching(idx))
            cov=self.covisibility_graph[idx]&set(self.cameras)
            score+=len(cov)*10
            scores[idx]=score
        return max(scores,key=scores.get) if scores else None

    def optimized_pnp(self, img_idx: int):
        obj,img,ids=[],[],[]
        for ref in self.cameras:
            key=(min(ref,img_idx),max(ref,img_idx))
            if key not in self.matches: continue
            for m in self.matches[key]:
                pt3d=self.points_3d.get(len(obj))
                if pt3d and ref in pt3d.observations and pt3d.observations[ref]==m.queryIdx:
                    obj.append(pt3d.xyz); img.append(self.features[img_idx][0][m.trainIdx].pt); ids.append(m)
        # add guided
        for pid,kp in self.fast_guided_matching(img_idx):
            obj.append(self.points_3d[pid].xyz); img.append(self.features[img_idx][0][kp].pt); ids.append(pid)
        if len(obj)<6: return None
        obj=np.array(obj,dtype=np.float32); img=np.array(img,dtype=np.float32)
        succ,rvec,tvec,inl=cv2.solvePnPRansac(obj,img,self.K_init,None,iterationsCount=500,
                                              reprojectionError=self.pnp_threshold,
                                              flags=cv2.SOLVEPNP_EPNP)
        if succ and inl is not None and len(inl)>=15:
            R,_=cv2.Rodrigues(rvec)
            return R,tvec.flatten(),inl.flatten().tolist()
        return None

    def fast_filter_bad_points(self):
        bad=[]
        for pid,pt in list(self.points_3d.items())[:self.batch_size]:
            if len(pt.observations)<self.min_track_length:
                bad.append(pid); continue
            errs=[]
            for cam_idx,kp_idx in pt.observations.items():
                cam=self.cameras.get(cam_idx)
                if cam:
                    errs.append(fast_reprojection_error(pt.xyz,cam.R,cam.t,cam.K,
                                                        np.array(self.features[cam_idx][0][kp_idx].pt)))
            if not errs or np.mean(errs)>self.max_reprojection_error:
                bad.append(pid)
        for pid in bad:
            self.points_3d.pop(pid,None)
            self.descriptor_cache.pop(pid,None)

    def initialize_reconstruction(self, init_pair: Tuple[int,int]) -> bool:
        i,j=init_pair
        key=(min(i,j),max(i,j))
        if key not in self.matches or len(self.matches[key])<50: return False
        m=self.matches[key];KP1,DS1=self.features[i];KP2,DS2=self.features[j]
        pts1=np.array([KP1[o.queryIdx].pt for o in m]); pts2=np.array([KP2[o.trainIdx].pt for o in m])
        F,mask=cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
        if F is None: return False
        E=self.K_init.T@F@self.K_init;inl=mask.ravel().astype(bool)
        pi, pj=pts1[inl],pts2[inl]
        _,R,t,_=cv2.recoverPose(E,pi,pj,self.K_init)
        self.cameras[i]=Camera(np.eye(3),np.zeros(3),self.K_init.copy());
        self.cameras[j]=Camera(R,t.flatten(),self.K_init.copy())
        self.covisibility_graph.clear(); 
        for idx,obs in enumerate(np.where(inl)[0]):
            pt3d=cv2.triangulatePoints(self.K_init@np.hstack([np.eye(3),np.zeros((3,1))]),
                                        self.K_init@np.hstack([R,t]),
                                        pts1[inl].T, pts2[inl].T)[:3].T
            if pt3d[idx,2]>0:
                self.points_3d[idx]=Point3D(pt3d[idx],np.array([128,128,128]),0.0,
                                              {i:m[obs].queryIdx,j:m[obs].trainIdx},
                                              DS1[m[obs].queryIdx])
                self.descriptor_cache[idx]=DS1[m[obs].queryIdx]
        return bool(self.points_3d)

    def estimate_initial_pose(self, img_idx:int) -> Optional[Tuple[np.ndarray,np.ndarray]]:
        best=[];ref=None
        for c in self.covisibility_graph[img_idx]&set(self.cameras):
            k=(min(c,img_idx),max(c,img_idx));
            if k in self.matches and len(self.matches[k])>20 and len(self.matches[k])>len(best):
                best,self_ref=self.matches[k],c;ref=c
        if ref is None: return None
        KP1,_=self.features[ref];KP2,_=self.features[img_idx]
        pts1=np.array([KP1[m.queryIdx].pt for m in best]);pts2=np.array([KP2[m.trainIdx].pt for m in best])
        F,mask=cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
        if F is None: return None
        E=self.K_init.T@F@self.K_init;inl=mask.ravel().astype(bool)
        _,Rr,tr,_=cv2.recoverPose(E,pts1[inl],pts2[inl],self.K_init)
        cam=self.cameras[ref]
        Rw=Rr@cam.R;tw=Rr@cam.t+tr.flatten()
        return Rw,tw

    def accelerated_reconstruct(self, features, matches, init_pair):
        self.features, self.matches = features, matches
        self.preprocess_features()
        if not self.initialize_reconstruction(init_pair): return False
        it=0;maxit=min(50,len(self.features)*2)
        while len(self.cameras)<len(self.features):
            it+=1;cand=self.smart_candidate_selection()
            if cand is None: break
            p=self.optimized_pnp(cand)
            if not p: self.failed_images.add(cand); continue
            R,t,inl=p
            self.cameras[cand]=Camera(R,t,self.K_init.copy(),True)
            self.parallel_triangulation(cand)
            if it%3==0: self.fast_filter_bad_points()
            if it>=maxit or len(self.cameras)/len(self.features)>self.early_termination_ratio: break
        return True
