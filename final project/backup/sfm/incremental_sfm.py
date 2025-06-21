import numpy as np
import cv2
from collections import defaultdict, deque
import heapq
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from numba import njit
from sklearn.neighbors import NearestNeighbors
import time
import faiss


@dataclass
class Point3D:
    """三维点数据结构"""
    xyz: np.ndarray  # 3D坐标
    color: np.ndarray  # RGB颜色
    error: float  # 重投影误差
    observations: Dict[int, int]  # {camera_id: keypoint_idx}
    descriptor: np.ndarray  # 特征描述子

@dataclass
class Camera:
    """相机数据结构"""
    R: np.ndarray  # 旋转矩阵
    t: np.ndarray  # 平移向量
    K: np.ndarray  # 内参矩阵
    registered: bool = False
    
class IncrementalSfM:
    def __init__(self, K_init, optimize_intrinsics=True,num_workers: int=8):
        self.K_init = K_init.copy()
        self.optimize_intrinsics = optimize_intrinsics
        self.num_workers = num_workers
        
        # 核心数据结构
        self.cameras = {}  # {img_idx: Camera}
        self.points_3d = {}  # {point_id: Point3D}
        self.features = {}  # {img_idx: (keypoints, descriptors)}
        self.matches = {}  # {(i,j): matches}
        
        # 候选队列和优先级管理
        self.candidate_queue = []  # 优先队列：(-score, img_idx)
        self.registered_cams = set()
        self.failed_images = set()
        
        # 倒排索引用于快速检索
        self.point_to_descriptor = {}  # {point_id: descriptor}
        self.descriptor_to_points = defaultdict(list)  # {desc_hash: [point_ids]}
        
        # 参数设置
        self.pnp_threshold = 4.0
        self.triangulation_threshold = 2.0
        self.min_triangulation_angle = 5.0  # 度
        self.max_reprojection_error = 10.0
        self.min_track_length = 2
        
    def adaptive_threshold_schedule(self, iteration: int, base_threshold: float) -> float:
        """自适应阈值调整"""
        # 早期迭代使用较松的阈值，后期收紧
        factor = max(0.5, 1.0 - iteration * 0.1)
        return base_threshold * factor

    def compute_candidate_score_fast(self, img_idx: int) -> float:
        if not self.points_3d or img_idx in self.registered_cams:
            return 0.0

        # 1) 构造全局描述子矩阵 P 和范数 normP（只要做一次就缓存）
        if not hasattr(self, "_P") or img_idx == 0:
            descs = [self.point_to_descriptor[pid] for pid in sorted(self.point_to_descriptor)]
            self._P = np.stack(descs, axis=0)       # (P, D)
            self._normP = np.linalg.norm(self._P, axis=1)  # (P,)

        # 2) 候选图像的描述子 F 和范数 normF
        kp, desc = self.features[img_idx]
        F = np.stack(desc, axis=0)                # (N, D)
        normF = np.linalg.norm(F, axis=1)         # (N,)

        # 3) 批量点积 + 归一化
        sims = F @ self._P.T                      # (N, P)
        sims /= normF[:,None] * self._normP[None,:]

        # 4) 取每个 3D 点的最大响应，做阈值并累加
        max_per_point = sims.max(axis=0)          # (P,)
        mask = max_per_point > 0.7
        score = max_per_point[mask].sum()

        # 5) 再加上共视权重
        for cam_idx in self.registered_cams:
            key = (min(img_idx, cam_idx), max(img_idx, cam_idx))
            if key in self.matches:
                score += len(self.matches[key]) * 0.1

        return score

    def compute_candidate_score(self, img_idx: int) -> float:
        """计算候选图像的优先级分数"""
        if not self.points_3d or img_idx in self.registered_cams:
            return 0.0
            
        score = 0.0
        kp, desc = self.features[img_idx]
        
        # 统计与已有3D点的潜在匹配数
        for point_id, pt3d in self.points_3d.items():
            # 使用描述子相似度快速估计匹配可能性
            if point_id in self.point_to_descriptor:
                ref_desc = self.point_to_descriptor[point_id]
                for i, d in enumerate(desc):
                    similarity = np.dot(ref_desc, d) / (np.linalg.norm(ref_desc) * np.linalg.norm(d))
                    if similarity > 0.7:  # 阈值可调
                        score += similarity
                        break
        
        # 加权已注册相机的共视程度
        for cam_idx in self.registered_cams:
            if (min(img_idx, cam_idx), max(img_idx, cam_idx)) in self.matches:
                matches = self.matches[(min(img_idx, cam_idx), max(img_idx, cam_idx))]
                score += len(matches) * 0.1
                
        return score

    def build_faiss_index(self):
        descs = [self.point_to_descriptor[pid] for pid in sorted(self.point_to_descriptor)]
        P = np.stack(descs, axis=0).astype('float32')    # (P, D)
        faiss.normalize_L2(P)
        self._index = faiss.IndexFlatIP(P.shape[1])
        self._index.add(P)
        self._point_ids = sorted(self.point_to_descriptor)

    def compute_candidate_score_faiss(self, img_idx: int) -> float:
        if not self.points_3d or img_idx in self.registered_cams:
            return 0.0
        if not hasattr(self, "_index"):
            self.build_faiss_index()

        kp, desc = self.features[img_idx]
        F = np.stack(desc, axis=0).astype('float32')      # (N, D)
        faiss.normalize_L2(F)
        sims, ids = self._index.search(F, 1)             # sims: (N,1), ids: (N,1)

        # threshold & 去重
        mask = sims[:,0] > 0.7
        unique_pids = set(self._point_ids[i] for i in ids[mask,0])
        score = sims[mask,0].sum()

        # 加权共视
        for cam_idx in self.registered_cams:
            key = (min(img_idx, cam_idx), max(img_idx, cam_idx))
            if key in self.matches:
                score += len(self.matches[key]) * 0.1

        return score

    def update_candidate_queue(self):
        """更新候选图像队列，跳过已注册和已失败的图像"""
        # 清空旧队列
        self.candidate_queue.clear()

        for img_idx in range(len(self.features)):
            # 跳过已经注册或已经标记失败的图像
            if img_idx in self.registered_cams or img_idx in self.failed_images:
                continue
            score = self.compute_candidate_score_faiss(img_idx)
            # 用负分数使得 heapq 实现最大堆行为
            heapq.heappush(self.candidate_queue, (-score, img_idx))
            


    def guided_matching(self, img_idx: int, search_radius: float = 50.0) -> List[Tuple[int, int]]:
        """引导式匹配：将3D点投影到新图像，在邻域内搜索匹配"""
        if img_idx in self.registered_cams:
            return []
            
        kp, desc = self.features[img_idx]
        kp_array = np.array([k.pt for k in kp])
        
        matches_2d3d = []
        
        # 估计初始位姿用于投影（可以用PnP的粗略结果）
        try:
            initial_pose = self.estimate_initial_pose(img_idx)
            if initial_pose is None:
                return []
            R_init, t_init = initial_pose
            # 确保t_init是正确的形状
            if t_init.ndim == 2:
                t_init = t_init.flatten()
        except:
            return []
        
        for point_id, pt3d in self.points_3d.items():
            try:
                # 投影3D点到当前图像
                xyz_cam = R_init @ pt3d.xyz + t_init
                
                # 检查点是否在相机前面 - 修复数组比较问题
                if xyz_cam.shape[0] > 1:
                    z_coord = xyz_cam[2]
                else:
                    z_coord = xyz_cam
                    
                if z_coord <= 0:  # 点在相机后面
                    continue
                    
                uv_proj = self.K_init @ xyz_cam
                if abs(uv_proj[2]) < 1e-8:  # 避免除零
                    continue
                    
                uv_proj = uv_proj[:2] / uv_proj[2]
                
                # 检查投影点是否在图像范围内
                if uv_proj[0] < 0 or uv_proj[1] < 0:
                    continue
                
                # 在投影点邻域内搜索最佳匹配
                distances = np.linalg.norm(kp_array - uv_proj.reshape(1, -1), axis=1)
                nearby_indices = np.where(distances < search_radius)[0]
                
                if len(nearby_indices) == 0:
                    continue
                    
                # 在候选点中找描述子最相似的
                ref_desc = self.point_to_descriptor.get(point_id)
                if ref_desc is None:
                    continue
                    
                best_match_idx = None
                best_similarity = 0.0
                
                for idx in nearby_indices:
                    # 确保描述子是归一化的
                    desc_norm = np.linalg.norm(desc[idx])
                    ref_desc_norm = np.linalg.norm(ref_desc)
                    
                    if desc_norm > 0 and ref_desc_norm > 0:
                        similarity = np.dot(ref_desc, desc[idx]) / (ref_desc_norm * desc_norm)
                        if similarity > best_similarity and similarity > 0.8:
                            best_similarity = similarity
                            best_match_idx = idx
                
                if best_match_idx is not None:
                    matches_2d3d.append((point_id, best_match_idx))
                    
            except Exception as e:
                # 跳过有问题的点
                continue
        
        return matches_2d3d
    
    def estimate_initial_pose(self, img_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """估计图像的初始位姿"""
        # 找到与已注册相机的匹配
        best_matches = []
        best_ref_cam = None
        
        for ref_cam in self.registered_cams:
            key = (min(img_idx, ref_cam), max(img_idx, ref_cam))
            if key in self.matches and len(self.matches[key]) > 20:
                if len(self.matches[key]) > len(best_matches):
                    best_matches = self.matches[key]
                    best_ref_cam = ref_cam
        
        if best_ref_cam is None:
            return None
            
        # 使用基础矩阵估计相对位姿
        kp1, _ = self.features[best_ref_cam]
        kp2, _ = self.features[img_idx]
        
        pts1 = np.array([kp1[m.queryIdx].pt for m in best_matches])
        pts2 = np.array([kp2[m.trainIdx].pt for m in best_matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K_init, method=cv2.RANSAC)
        if E is None:
            return None
            
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K_init)
        
        # 转换到世界坐标系
        ref_cam = self.cameras[best_ref_cam]
        R_world = R @ ref_cam.R
        # 确保t和ref_cam.t都是一维向量
        t_flat = t.flatten() if t.ndim > 1 else t
        ref_t_flat = ref_cam.t.flatten() if ref_cam.t.ndim > 1 else ref_cam.t
        t_world = R @ ref_t_flat + t_flat
        
        return R_world, t_world
    
    def multiview_triangulation(self, img_idx: int):
        """多视图三角化：新相机与所有可共视的已注册相机做三角化"""
        if img_idx not in self.registered_cams:
            return
            
        new_points = {}
        
        # 遍历所有已注册的相机
        for ref_idx in self.registered_cams:
            if ref_idx == img_idx:
                continue
                
            key = (min(img_idx, ref_idx), max(img_idx, ref_idx))
            if key not in self.matches:
                continue
                
            matches = self.matches[key]
            if len(matches) < 10:
                continue
                
            # 准备三角化数据
            kp1, _ = self.features[ref_idx]
            kp2, _ = self.features[img_idx]
            
            cam1 = self.cameras[ref_idx]
            cam2 = self.cameras[img_idx]
            
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
            
            # 构造投影矩阵
            P1 = cam1.K @ np.hstack([cam1.R, cam1.t.reshape(-1, 1)])
            P2 = cam2.K @ np.hstack([cam2.R, cam2.t.reshape(-1, 1)])
            
            # 三角化点
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            points_3d = points_4d[:3] / points_4d[3]
            
            # 几何验证和误差筛除
            for i, (m, pt3d) in enumerate(zip(matches, points_3d.T)):
                # 检查三角化角度
                ray1 = pt3d - cam1.t
                ray2 = pt3d - cam2.t
                angle = np.arccos(np.clip(np.dot(ray1, ray2) / 
                                        (np.linalg.norm(ray1) * np.linalg.norm(ray2)), -1, 1))
                angle_deg = np.degrees(angle)
                
                if angle_deg < self.min_triangulation_angle:
                    continue
                
                # 检查重投影误差
                reproj_error1 = self.compute_reprojection_error(pt3d, cam1, kp1[m.queryIdx].pt)
                reproj_error2 = self.compute_reprojection_error(pt3d, cam2, kp2[m.trainIdx].pt)
                
                if reproj_error1 > self.max_reprojection_error or reproj_error2 > self.max_reprojection_error:
                    continue
                
                # 创建新的3D点
                point_id = len(self.points_3d) + len(new_points)
                observations = {ref_idx: m.queryIdx, img_idx: m.trainIdx}
                
                # 计算颜色（简单平均）
                # 这里需要图像数据来提取颜色，暂时用默认值
                color = np.array([128, 128, 128])
                
                # 获取描述子（使用第一个观测的描述子）
                _, desc1 = self.features[ref_idx]
                descriptor = desc1[m.queryIdx]
                
                new_points[point_id] = Point3D(
                    xyz=pt3d,
                    color=color,
                    error=max(reproj_error1, reproj_error2),
                    observations=observations,
                    descriptor=descriptor
                )
        
        # 添加新点到全局结构
        self.points_3d.update(new_points)
        
        # 更新倒排索引
        for point_id, pt3d in new_points.items():
            self.point_to_descriptor[point_id] = pt3d.descriptor
    
    def parallel_triangulation(self, img_idx: int):
        """
        并行三角化：对新注册相机与所有已注册相机做三角化。
        """
        if img_idx not in self.registered_cams:
            return

        def triangulate_pair(ref_idx: int) -> Dict[int, Point3D]:
            new_pts: Dict[int, Point3D] = {}
            key = (min(img_idx, ref_idx), max(img_idx, ref_idx))
            matches = self.matches.get(key, [])
            if len(matches) < self.min_track_length:
                return {}

            cam1 = self.cameras[ref_idx]
            cam2 = self.cameras[img_idx]
            kp1, _ = self.features[ref_idx]
            kp2, _ = self.features[img_idx]

            # 构造投影矩阵
            P1 = cam1.K @ np.hstack([cam1.R, cam1.t.reshape(-1,1)])
            P2 = cam2.K @ np.hstack([cam2.R, cam2.t.reshape(-1,1)])
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches]).T
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches]).T

            # 三角化
            pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
            pts3d = (pts4d[:3] / pts4d[3]).T  # N×3

            for i, pt3 in enumerate(pts3d):
                m = matches[i]
                # 检查视差角
                ray1 = pt3 - cam1.t.flatten()
                ray2 = pt3 - cam2.t.flatten()
                cos_angle = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2) + 1e-8)
                if np.degrees(np.arccos(np.clip(cos_angle, -1, 1))) < self.min_triangulation_angle:
                    continue
                # 检查重投影误差
                err1 = self.compute_reprojection_error(pt3, cam1, kp1[m.queryIdx].pt)
                err2 = self.compute_reprojection_error(pt3, cam2, kp2[m.trainIdx].pt)
                if err1 > self.max_reprojection_error or err2 > self.max_reprojection_error:
                    continue

                pid = len(self.points_3d) + len(new_pts)
                obs = {ref_idx: m.queryIdx, img_idx: m.trainIdx}
                desc = self.features[ref_idx][1][m.queryIdx]
                new_pts[pid] = Point3D(xyz=pt3, color=np.array([128,128,128]), error=max(err1,err2),
                                      observations=obs, descriptor=desc)
            return new_pts

        # 并行执行所有参考相机
        combined: Dict[int, Point3D] = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as exe:
            futures = [exe.submit(triangulate_pair, ref) for ref in self.registered_cams if ref != img_idx]
            for f in futures:
                combined.update(f.result())

        # 更新全局
        self.points_3d.update(combined)
        for pid, pt in combined.items():
            self.point_to_descriptor[pid] = pt.descriptor    

    def compute_reprojection_error(self, pt3d: np.ndarray, camera: Camera, observed_pt: Tuple[float, float]) -> float:
        """计算重投影误差"""
        try:
            # 确保向量维度正确
            t_cam = camera.t.flatten() if camera.t.ndim > 1 else camera.t
            xyz_cam = camera.R @ pt3d + t_cam
            
            # 检查点是否在相机前面
            if xyz_cam.shape[0] > 1:
                z_coord = xyz_cam[2]
            else:
                z_coord = xyz_cam
                
            if z_coord <= 0:
                return float('inf')
                
            uv_proj = camera.K @ xyz_cam
            if abs(uv_proj[2]) < 1e-8:
                return float('inf')
                
            uv_proj = uv_proj[:2] / uv_proj[2]
            
            return np.linalg.norm(uv_proj - np.array(observed_pt))
        except:
            return float('inf')
    
    def ransac_pnp_with_adaptive_threshold(self, img_idx: int, iteration: int) -> Optional[Tuple[np.ndarray, np.ndarray, List[int]]]:
        """带自适应阈值的RANSAC PnP"""
        # 收集2D-3D对应点
        obj_pts, img_pts, point_ids = self.collect_2d3d_correspondences(img_idx)
        
        if len(obj_pts) < 6:
            return None
        
        obj_pts = np.array(obj_pts)
        img_pts = np.array(img_pts)
        
        # 自适应阈值
        threshold = self.adaptive_threshold_schedule(iteration, self.pnp_threshold)
        
        # 多次不同阈值的RANSAC
        thresholds = [threshold, threshold * 0.8, threshold * 1.2]
        best_result = None
        best_inlier_count = 0
        
        for thresh in thresholds:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, self.K_init, None,
                reprojectionError=thresh,
                iterationsCount=1000,
                confidence=0.90
            )
            
            if success and inliers is not None:
                inlier_count = len(inliers)
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    R, _ = cv2.Rodrigues(rvec)
                    best_result = (R, tvec.flatten(), inliers.flatten().tolist())
        
        return best_result
    
    def collect_2d3d_correspondences(self, img_idx: int) -> Tuple[List, List, List]:
        """收集2D-3D对应点（包括引导式匹配的结果）"""
        obj_pts, img_pts, point_ids = [], [], []
        
        # 检查输入有效性
        if img_idx not in self.features:
            return obj_pts, img_pts, point_ids
        
        # 1. 从现有匹配中收集
        for ref_cam in self.registered_cams:
            key = (min(img_idx, ref_cam), max(img_idx, ref_cam))
            if key not in self.matches:
                continue
                
            matches = self.matches[key]
            if not matches or len(matches) == 0:
                continue
                
            # 检查特征是否存在
            if ref_cam not in self.features or img_idx not in self.features:
                continue
                
            kp_ref, _ = self.features[ref_cam]
            kp_new, _ = self.features[img_idx]
            
            for m in matches:
                try:
                    # 查找对应的3D点
                    for point_id, pt3d in self.points_3d.items():
                        if ref_cam in pt3d.observations:
                            ref_kp_idx = pt3d.observations[ref_cam]
                            if ref_kp_idx == m.queryIdx:
                                # 检查索引有效性
                                if (m.trainIdx < len(kp_new) and 
                                    ref_kp_idx < len(kp_ref) and
                                    hasattr(pt3d, 'xyz') and
                                    pt3d.xyz is not None):
                                    
                                    obj_pts.append(pt3d.xyz)
                                    img_pts.append(kp_new[m.trainIdx].pt)
                                    point_ids.append(point_id)
                                break
                except (IndexError, AttributeError) as e:
                    continue
        
        # 2. 添加引导式匹配的结果
        try:
            guided_matches = self.guided_matching(img_idx)
            if guided_matches and img_idx in self.features:
                kp_new, _ = self.features[img_idx]
                
                for point_id, kp_idx in guided_matches:
                    try:
                        if (point_id in self.points_3d and 
                            kp_idx < len(kp_new) and
                            hasattr(self.points_3d[point_id], 'xyz') and
                            self.points_3d[point_id].xyz is not None):
                            
                            obj_pts.append(self.points_3d[point_id].xyz)
                            img_pts.append(kp_new[kp_idx].pt)
                            point_ids.append(point_id)
                    except (IndexError, AttributeError, KeyError):
                        continue
        except Exception as e:
            # 如果引导式匹配失败，至少返回从现有匹配中收集的结果
            print(f"引导式匹配失败，图像 {img_idx}: {e}")
        
        return obj_pts, img_pts, point_ids
    
    def local_bundle_adjustment(self, new_cam_idx: int, window_size: int = 10):
        """局部BA：优化新相机及其共视的相机和点"""
        # 收集参与BA的相机和点
        ba_cameras = {new_cam_idx}
        ba_points = set()
        
        # 找到与新相机共视的点
        for point_id, pt3d in self.points_3d.items():
            if new_cam_idx in pt3d.observations:
                ba_points.add(point_id)
                # 添加观测到这些点的其他相机
                for cam_idx in pt3d.observations.keys():
                    if cam_idx != new_cam_idx and len(ba_cameras) < window_size:
                        ba_cameras.add(cam_idx)
        
        # 这里应该调用BA优化器（如g2o, ceres等）
        # 简化版本：只更新内参（如果允许）
        if self.optimize_intrinsics and len(ba_cameras) > 2:
            self.refine_intrinsics(ba_cameras, ba_points)
    
    def refine_intrinsics(self, camera_indices: Set[int], point_indices: Set[int]):
        """暂时跳过内参优化，使用固定的内参矩阵"""
        print("Using fixed intrinsic parameters, skipping refinement")
        return
    
    def reconstruct_incremental(self, features: Dict, matches: Dict, init_pair: Tuple[int, int]):
        """主要的增量重建函数"""
        self.features = features
        self.matches = matches
        
        # 1. 初始化：使用两视图重建
        success = self.initialize_reconstruction(init_pair)
        if not success:
            print("初始化失败")
            return False
        
        print(f"初始化成功，注册了相机 {init_pair[0]} 和 {init_pair[1]}")
        print(f"初始3D点数量: {len(self.points_3d)}")
        
        # 2. 增量注册其余相机
        iteration = 0
        while len(self.registered_cams) < len(self.features):
            iteration += 1
            
            # 更新候选队列
            t0=time.time()
            self.update_candidate_queue()
            t1=time.time()
            print(f"更新候选队列耗时: {t1-t0:.2f}秒")
            
            if not self.candidate_queue:
                print("没有更多候选图像")
                break
            
            # 选择下一个最佳候选
            _, next_img_idx = heapq.heappop(self.candidate_queue)
            
            print(f"\n第 {iteration} 次迭代：尝试注册图像 {next_img_idx}")
            
            # PnP位姿估计
            pnp_result = self.ransac_pnp_with_adaptive_threshold(next_img_idx, iteration)
            t2 = time.time()
            print(f"PnP估计耗时: {t2-t1:.2f}秒")
            
            if pnp_result is None:
                print(f"图像 {next_img_idx} PnP失败")
                self.failed_images.add(next_img_idx)
                continue
            
            R, t, inliers = pnp_result
            
            # 验证位姿质量
            if len(inliers) < 15:  # 最少内点数
                print(f"图像 {next_img_idx} 内点数过少: {len(inliers)}")
                self.failed_images.add(next_img_idx)
                continue
            
            # 注册新相机
            if self.optimize_intrinsics:
                K_cam = self.K_init.copy()  # 每个相机可以有独立内参
            else:
                K_cam = self.K_init
                
            self.cameras[next_img_idx] = Camera(R=R, t=t, K=K_cam, registered=True)
            self.registered_cams.add(next_img_idx)
            
            print(f"成功注册图像 {next_img_idx}，内点数: {len(inliers)}")
            
            # 多视图三角化
            old_point_count = len(self.points_3d)
            # self.multiview_triangulation(next_img_idx)
            self.parallel_triangulation(next_img_idx)
            new_point_count = len(self.points_3d) - old_point_count
            t3 = time.time()
            print(f"多视图三角化耗时: {t3-t2:.2f}秒")
            print(f"新增3D点: {new_point_count}，总点数: {len(self.points_3d)}")
            
            # 局部Bundle Adjustment
            self.local_bundle_adjustment(next_img_idx)
            
            # 每隔几次迭代做一次全局清理
            if iteration % 5 == 0:
                self.filter_bad_points()
                
            # if iteration > 70:
            #     print("迭代次数超过50，停止增量重建")
            #     break
        
        print(f"\n重建完成！")
        print(f"注册相机数: {len(self.registered_cams)}/{len(self.features)}")
        print(f"3D点数: {len(self.points_3d)}")
        
        return True
    
    def initialize_reconstruction(self, init_pair: Tuple[int, int]) -> bool:
        """两视图初始化"""
        i, j = init_pair
        
        if (i, j) not in self.matches and (j, i) not in self.matches:
            return False
        
        key = (min(i, j), max(i, j))
        matches = self.matches[key]
        
        if len(matches) < 50:
            return False
        
        # 获取特征点
        kp_i, desc_i = self.features[i]
        kp_j, desc_j = self.features[j]
        
        pts_i = np.array([kp_i[m.queryIdx].pt for m in matches])
        pts_j = np.array([kp_j[m.trainIdx].pt for m in matches])
        
        # 估计基础矩阵
        F, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC)
        if F is None:
            return False
        
        # 估计本质矩阵
        E = self.K_init.T @ F @ self.K_init
        
        # 恢复位姿
        inlier_mask = mask.ravel().astype(bool)
        pts_i_inlier = pts_i[inlier_mask]
        pts_j_inlier = pts_j[inlier_mask]
        
        _, R, t, _ = cv2.recoverPose(E, pts_i_inlier, pts_j_inlier, self.K_init)
        
        # 设置第一个相机为原点
        self.cameras[i] = Camera(R=np.eye(3), t=np.zeros(3), K=self.K_init.copy())
        # 确保t是一维向量
        t_flat = t.flatten() if t.ndim > 1 else t
        self.cameras[j] = Camera(R=R, t=t_flat, K=self.K_init.copy())
        
        self.registered_cams.add(i)
        self.registered_cams.add(j)
        
        # 三角化初始点
        P1 = self.K_init @ np.hstack([np.eye(3), np.zeros(3).reshape(-1, 1)])
        P2 = self.K_init @ np.hstack([R, t])
        
        points_4d = cv2.triangulatePoints(P1, P2, pts_i_inlier.T, pts_j_inlier.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        # 添加有效的3D点
        inlier_indices = np.where(inlier_mask)[0]
        for idx, (pt3d, orig_idx) in enumerate(zip(points_3d.T, inlier_indices)):
            # 基本几何验证
            if len(pt3d.shape) > 0 and pt3d.shape[0] >= 3:
                z_coord = pt3d[2] if pt3d.ndim == 1 else pt3d[2]
                if z_coord > 0:  # 在相机前方
                    match = matches[orig_idx]
                    observations = {i: match.queryIdx, j: match.trainIdx}
                    
                    point_id = len(self.points_3d)
                    self.points_3d[point_id] = Point3D(
                        xyz=pt3d,
                        color=np.array([128, 128, 128]),
                        error=0.0,
                        observations=observations,
                        descriptor=desc_i[match.queryIdx]
                    )
                    
                    # 更新倒排索引
                    self.point_to_descriptor[point_id] = desc_i[match.queryIdx]
        
        return len(self.points_3d) > 0
    
    def filter_bad_points(self):
        """过滤质量差的3D点"""
        bad_points = []
        
        for point_id, pt3d in self.points_3d.items():
            # 检查观测数量
            if len(pt3d.observations) < self.min_track_length:
                bad_points.append(point_id)
                continue
            
            # 检查重投影误差
            total_error = 0.0
            valid_observations = 0
            
            for cam_idx, kp_idx in pt3d.observations.items():
                if cam_idx in self.cameras:
                    camera = self.cameras[cam_idx]
                    kp, _ = self.features[cam_idx]
                    
                    error = self.compute_reprojection_error(pt3d.xyz, camera, kp[kp_idx].pt)
                    if error < self.max_reprojection_error:
                        total_error += error
                        valid_observations += 1
            
            if valid_observations == 0 or total_error / valid_observations > self.max_reprojection_error:
                bad_points.append(point_id)
        
        # 删除坏点
        for point_id in bad_points:
            if point_id in self.points_3d:
                del self.points_3d[point_id]
            if point_id in self.point_to_descriptor:
                del self.point_to_descriptor[point_id]
        
        if bad_points:
            print(f"清理了 {len(bad_points)} 个质量差的3D点")
    
    
    
    
    # def refine_intrinsics(self, camera_indices: Set[int], point_indices: Set[int]):
    #     obj_pts_per_view = []
    #     img_pts_per_view = []

    #     # 对每个相机单独收集它的 3D-2D 对
    #     for cam_idx in camera_indices:
    #         if cam_idx not in self.cameras or cam_idx >= len(self.features):
    #             continue

    #         camera = self.cameras[cam_idx]
    #         kp_list, _ = self.features[cam_idx]

    #         pts3d_list = []
    #         pts2d_list = []
    #         for pid in point_indices:
    #             if pid not in self.points_3d:
    #                 continue
    #             obs = self.points_3d[pid].observations
    #             if cam_idx not in obs:
    #                 continue

    #             kp_idx = obs[cam_idx]
    #             # 世界坐标 → 相机坐标
    #             xyz_cam = camera.R @ self.points_3d[pid].xyz + camera.t
    #             pts3d_list.append(xyz_cam)
    #             pts2d_list.append(kp_list[kp_idx].pt)

    #         if len(pts3d_list) < 6:
    #             # 单个视角至少需要几个点才能做标定
    #             continue

    #         # 转为 (N,1,3) 和 (N,1,2)，并强制 float32
    #         obj_np = np.asarray(pts3d_list, dtype=np.float32).reshape(-1,1,3)
    #         img_np = np.asarray(pts2d_list, dtype=np.float32).reshape(-1,1,2)

    #         obj_pts_per_view.append(obj_np)
    #         img_pts_per_view.append(img_np)

    #     if len(obj_pts_per_view) < 2:
    #         # 至少要两个视角才能优化内参
    #         return

    #     # 真实的图像分辨率，例 (width, height)
    #     image_size = (4032, 3024)

    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    #     ret, K_refined, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    #         obj_pts_per_view,
    #         img_pts_per_view,
    #         image_size,
    #         self.K_init,
    #         None,
    #         criteria=criteria
    #     )

    #     # ret 是重投影误差（RMS），总是大于 0，所以不用做 if ret: 这种布尔检查
    #     alpha = 0.1
    #     self.K_init = (1 - alpha) * self.K_init + alpha * K_refined
# 使用示例
def example_usage():
    """使用示例"""
    # 假设已有的数据
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    
    # 初始化SfM系统
    sfm = IncrementalSfM(K, optimize_intrinsics=True)
    
    # 假设的特征和匹配数据
    features = {}  # {img_idx: (keypoints, descriptors)}
    matches = {}   # {(i,j): matches}
    
    # 执行重建
    # success = sfm.reconstruct_incremental(features, matches, init_pair=(0, 1))
    
    return sfm


if __name__ == "__main__":
    sfm_system = example_usage()
    print("改进的增量式SfM系统已创建")