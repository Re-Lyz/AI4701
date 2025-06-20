import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any,Tuple
from scipy.spatial.transform import Rotation as R_scipy
import open3d as o3d
from contextlib import redirect_stdout
from feature_extraction import load_or_extract_features
from feature_matching import load_or_match_image_pairs, filter_matches_by_homography
from initial_recon import estimate_pose, print_pose_info
from pnp_recon import pnp_pose, visualize_with_open3d, triangulate_points
from bundle_adjustment import BundleAdjustment

def compute_parallax(
    pts1: np.ndarray,
    pts2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray
) -> float:
    """
    计算一组匹配内点的平均归一化视差：
      1) 将像素点转换为归一化相机坐标系下的射线方向
      2) 将第一幅图的射线变换到第二幅图坐标系
      3) 计算两组射线的夹角（视差），并取平均
    """
    # 1. 转为齐次坐标并归一化
    ones = np.ones((pts1.shape[0], 1))
    homo1 = np.hstack([pts1, ones])
    homo2 = np.hstack([pts2, ones])

    rays1 = (np.linalg.inv(K) @ homo1.T).T
    rays2 = (np.linalg.inv(K) @ homo2.T).T

    # 2. 将第一张图的射线旋转平移到第二张图坐标系
    rays1_in_2 = (R @ rays1.T + t).T

    # 3. 归一化方向向量
    rays1_norm = rays1 / np.linalg.norm(rays1, axis=1, keepdims=True)
    rays2_norm = rays1_in_2 / np.linalg.norm(rays1_in_2, axis=1, keepdims=True)

    # 4. 计算夹角（点积的 arccos）
    cos_angles = np.sum(rays1_norm * rays2_norm, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    return float(np.mean(angles))

def pick_initial_pair(
    K: np.ndarray,
    al_matches: List[List[List[cv2.DMatch]]],
    keypoints: List[List[cv2.KeyPoint]],
    min_inliers: int = 50,
    parallax_thresh: float = 0.01
) -> Tuple[int, int, np.ndarray, np.ndarray, List[cv2.DMatch]]:
    """
    从 al_matches 中选出用于初始化的最佳图像对，
    并使用 RANSAC + cv2.recoverPose 恢复 R, t, 以及内点匹配 inlier_matches。

    Args:
        K: 相机内参矩阵
        al_matches: shape = [N][N] 的匹配列表矩阵，其中 al_matches[i][j] 是 i->j 的候选匹配
        keypoints: 每张图对应的 KeyPoint 列表
        min_inliers: 内点匹配最小阈值
        parallax_thresh: 归一化视差最小阈值

    Returns:
        best_i, best_j: 选中的图像对索引
        best_R, best_t: 恢复的相对位姿
        best_inliers: 对应的内点匹配列表
    """
    num_images = len(al_matches)
    best_score = 0
    best_i, best_j = -1, -1
    best_R = np.eye(3)
    best_t = np.zeros((3,1))
    best_inliers: List[cv2.DMatch] = []

    for i in range(num_images):
        for j in range(i+1, num_images):
            matches_ij = al_matches[i][j]
            if len(matches_ij) < min_inliers:
                continue

            # 提取匹配点坐标
            pts1 = np.array([keypoints[i][m.queryIdx].pt for m in matches_ij])
            pts2 = np.array([keypoints[j][m.trainIdx].pt for m in matches_ij])

            # 1) 用 RANSAC 估计本质矩阵
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            if E is None or mask is None:
                continue

            # 筛选足够多的内点
            inlier_mask = mask.ravel().astype(bool)
            n_inliers = int(inlier_mask.sum())
            if n_inliers < min_inliers:
                continue

            # 2) 恢复相对位姿
            _, R, t, mask_pose = cv2.recoverPose(
                E, pts1, pts2, K, mask=mask
            )
            pose_mask = mask_pose.ravel().astype(bool)

            # 收集真正的内点 matches
            inlier_matches = [
                m for m, ok in zip(matches_ij, pose_mask) if ok
            ]

            # 3) 计算平均视差
            inlier_pts1 = pts1[pose_mask]
            inlier_pts2 = pts2[pose_mask]
            parallax = compute_parallax(inlier_pts1, inlier_pts2, R, t, K)
            if parallax < parallax_thresh:
                continue

            # 4) 用内点数作为评分并更新最优
            if n_inliers > best_score:
                best_score = n_inliers
                best_i, best_j = i, j
                best_R, best_t = R, t
                best_inliers = inlier_matches

    if best_score == 0:
        raise RuntimeError("没有找到满足条件的初始图像对，请降低阈值或检查匹配质量。")

    return best_i, best_j, best_R, best_t, best_inliers
class ReconstructionPipeline:
    def __init__(
        self,
        image_folder: str,
        camera_intrinsics: str ,
        output_dir: str = 'output',
        feature_extractor: bool = True,
        feature_matcher: bool = True,
        initial_pose_estimator: bool = True,
        pnp_estimator: bool = True,
        bundle_adjustment: bool = True,
        verbose: bool = True,
        load: bool = False,                 
        save: bool = False,
    ):
        self.image_folder = image_folder    
        self.camera_intrinsics = camera_intrinsics
        self.output_dir = output_dir
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.initial_pose_estimator = initial_pose_estimator
        self.pnp_estimator = pnp_estimator
        self.bundle_adjustment = bundle_adjustment
        self.verbose = verbose
        self.load = load
        self.save = save

        # Gather image paths
        self.image_paths = []
        for root, dirs, files in os.walk(image_folder):
            self.image_paths = [os.path.join(root, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if self.image_paths:
                break
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in folder: {image_folder}")

        # Load camera intrinsics
        k_path = self.camera_intrinsics
        if not os.path.exists(k_path):
            raise FileNotFoundError(f"找不到相机内参文件：{k_path}")
        self.K = np.loadtxt(k_path, dtype=np.float32)
        if self.K.shape != (3, 3):
            raise ValueError(f"相机内参矩阵形状不正确，应为3x3，但得到 {self.K.shape}")

        # Placeholders
        self.features: List[Any] = []
        self.matches: List[Any] = []
        self.poses: List[Dict[str, Any]] = []
        self.points_3d: np.ndarray = None
        self.point_observations: Dict[int, Dict[int, tuple]] = {}
        self.al_matches: List[List[List[cv2.DMatch]]] = []
        
        
    def run(self):
        start_time = time.time()
        
        def _run_quiet(func, *args, **kwargs):
            if self.verbose:
                return func(*args, **kwargs)
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull):
                    return func(*args, **kwargs)
                
        # 1. Feature Extraction
        if self.feature_extractor:
            print("Extracting features from images...")
            self.features = _run_quiet(
                load_or_extract_features,
                self.image_paths,
                method='sift',
                save=self.save,
                load=self.load,
                output_dir=self.output_dir
            )

        # 2. Feature Matching
        if self.feature_matcher:
            print("Matching features between image pairs...")
            all_matches = _run_quiet(
                load_or_match_image_pairs,
                self.features,
                method='sift',
                save=self.save,
                load=self.load,
                output_dir=self.output_dir
            )
            n_images = len(self.features)
            
            # 这里后面可以加个消融实验 RANSAC过滤
            al_matches = []
            for i in range(n_images):
                row = []
                for j in range(n_images):
                    # 自身与自身不匹配
                    if i == j:
                        row.append([])
                        continue
                    
                    kp1, _ = self.features[i]
                    kp2, _ = self.features[j]
                    matches = all_matches[i][j]

                    # 少于 4 对匹配无法估计单应，直接返回空列表
                    if len(matches) < 4:
                        row.append([])
                        continue
                    
                    # 2. RANSAC + 单应性过滤
                    inlier_matches, _ = _run_quiet(filter_matches_by_homography, kp1, kp2, matches)
                    row.append(inlier_matches)
                al_matches.append(row)
            self.al_matches = al_matches
            self.matches = []
            n = len(self.features)
            for i in range(n):
                for j in range(i+1, n):
                    matches_ij = al_matches[i][j]
                    if len(matches_ij) >= 4:
                        self.matches.append((i, j, matches_ij))

        # 3. Initial Pose Estimation
        if self.initial_pose_estimator and self.matches:
            print("Estimating initial pose from first image pair...")
            i, j, R, t, inliers = pick_initial_pair(self.K, self.al_matches, keypoints=[kp_tuple[0] for kp_tuple in self.features])
            # 记录真正的参考图像索引
            self.init_i, self.init_j = i, j
            # 保存第一对相机：第 i 张为“0”，第 j 张为“1”
            self.poses = [
                {'R': np.eye(3),         't': np.zeros((3,1)), 'inliers': None,     'pair': (None, i)},
                {'R': R,                 't': t,               'inliers': inliers, 'pair': (i, j)},
            ]
            # 用于后续判断哪些图已加入
            self.registered_cams = [i, j]
            print_pose_info(R, t, len(inliers), len(inliers))

        # 4. PnP Pose Estimation and Incremental Triangulation
        if self.pnp_estimator and len(self.poses) >= 2:
            print("Triangulating initial 3D points between frames "
                  f"{self.init_i} & {self.init_j} ...")

            # 相机 0 是 identity；相机 1 是 (i→j) 的初始外参
            zero  = self.poses[0]
            first = self.poses[1]
            kp_i, _ = self.features[self.init_i]
            kp_j, _ = self.features[self.init_j]

            # 1) 初始化三角化得到 valid_idxs，对应的 self.points_3d[0:N_init]
            initial_pts3d, valid_idxs = triangulate_points(
                first['inliers'], kp_i, kp_j, self.K,
                zero['R'], zero['t'], first['R'], first['t']
            )
            self.points_3d = initial_pts3d.copy()

            # 2) 建立 queryIdx -> pid 的映射
            query_to_pid = {}
            for pid, valid_idx in enumerate(valid_idxs):
                m0 = first['inliers'][valid_idx]
                query_to_pid[m0.queryIdx] = pid

            # 3) 用列表存观测，最后再转 dict
            obs_list = [
                {
                    self.init_i: kp_i[first['inliers'][i].queryIdx].pt,
                    self.init_j: kp_j[first['inliers'][i].trainIdx].pt
                }
                for i in valid_idxs
            ]

            # 4) 增量处理每张新图，只和 init_i 做 PnP
            for img_idx in range(len(self.image_paths)):
                if img_idx in self.registered_cams:
                    continue
                
                # —— 必须先拿到这张图的 keypoints —— 
                kp_new, desc_new = self.features[img_idx]

                filt = self.al_matches[self.init_i][img_idx]
                if len(filt) < 6:
                    continue
                
                # 5) 批量构造 obj_pts, img_pts, pid_list
                obj_pts, img_pts, pid_list = [], [], []
                for m in filt:
                    pid = query_to_pid.get(m.queryIdx)
                    if pid is not None:
                        obj_pts.append(self.points_3d[pid])
                        img_pts.append(kp_new[m.trainIdx].pt)
                        pid_list.append(pid)

                if len(obj_pts) < 6:
                    continue
                
                # 6) RANSAC PnP
                Rn, tn, inliers = pnp_pose(img_pts, obj_pts, self.K)
                inlier_idxs = inliers.flatten().tolist()

                # 7) 保存新相机
                self.poses.append({
                    'R': Rn, 't': tn, 'inliers': inlier_idxs,
                    'pair': (self.init_i, img_idx)
                })
                self.registered_cams.append(img_idx)

                # 8) 批量更新 obs_list
                for idx in inlier_idxs:
                    pid = pid_list[idx]
                    obs_list[pid][img_idx] = img_pts[idx]

            # 9) 最终转回 dict 形式
            self.point_observations = {
                pid: obs for pid, obs in enumerate(obs_list)
            }


        
        camera_matrices = []
        for pose in self.poses:
            T = np.eye(4, dtype=np.float64)
            T[:3,:3] = pose['R']
            T[:3, 3] = pose['t'].reshape(3,)
            camera_matrices.append(T)
        # 直接保存这一份
        camera_matrices_to_save = camera_matrices
        
        # 5. Bundle Adjustment
        if self.bundle_adjustment:
            print("Running bundle adjustment...")

            # 5.1 将 self.poses 转为 4x4 矩阵列表
            camera_matrices = []
            for pose in self.poses:
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = pose['R']
                # 如果 t 的形状是 (3,1) 或 (3,)
                t_vec = pose['t'].reshape(3,)
                T[:3, 3] = t_vec
                camera_matrices.append(T)

            # 5.2 从 point_observations 构建观测数组
            observations, point_indices, camera_indices = [], [], []
            for pt_idx, cam_dict in self.point_observations.items():
                for cam_idx, xy in cam_dict.items():
                    observations.append(xy)          # (x, y)
                    point_indices.append(pt_idx)     # 该观测对应的三维点索引
                    camera_indices.append(cam_idx)   # 该观测来自的相机索引

            # 5.3 调用 BundleAdjustment.optimize
            ba = BundleAdjustment(camera_intrinsics=self.K, verbose=True)
            optimized_cams, optimized_pts3d, ba_info = ba.optimize(
                cameras=camera_matrices,
                points_3d=self.points_3d,
                observations=observations,
                point_indices=point_indices,
                camera_indices=camera_indices
            )

            # 5.4 将优化结果写回 self.poses 和 self.points_3d
            self.points_3d = optimized_pts3d
            camera_matrices_to_save = optimized_cams
            for idx, T_opt in enumerate(optimized_cams):
                R_opt = T_opt[:3, :3]
                t_opt = T_opt[:3, 3].reshape(3, 1)
                self.poses[idx]['R'] = R_opt
                self.poses[idx]['t'] = t_opt

            # 5.5 输出优化信息
            print(f"Bundle Adjustment 完成：")
            print(f"  Success      : {ba_info['success']}")
            print(f"  Initial cost : {ba_info['initial_cost']:.6f}")
            print(f"  Final cost   : {ba_info['final_cost']:.6f}")
            print(f"  Iterations   : {ba_info['iterations']}")

        output_folder = self.output_dir
        os.makedirs(output_folder, exist_ok=True)

        ply_path = os.path.join(output_folder, "optimized_point_cloud.ply")
        with open(ply_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {self.points_3d.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for pt in self.points_3d:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
        print(f"[INFO] Saved optimized point cloud to: {ply_path}")

        traj_path = os.path.join(output_folder, "optimized_camera_trajectory.txt")
        with open(traj_path, 'w') as f:
            f.write("# CamIdx tx ty tz qx qy qz qw\n")
            for i, T in enumerate(camera_matrices_to_save):
                pos = T[:3, 3]
                R_mat = T[:3, :3]
                quat = R_scipy.from_matrix(R_mat).as_quat()  # (x,y,z,w)
                f.write(
                    f"{i} "
                    f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
                )
        print(f"[INFO] Saved optimized camera trajectory to: {traj_path}")

        end_time = time.time()
        print(f"Reconstruction completed in {end_time - start_time:.2f} seconds.")

    def visualize_from_output(self):
        """
        从 output_dir 直接读取优化后的点云 (PLY) 和相机轨迹 (TXT)，
        然后调用 visualize_with_open3d 可视化，不需要重新跑 pipeline。
        """
        # --- 1. 读取点云 ---
        ply_file = os.path.join(self.output_dir, "optimized_point_cloud.ply")
        if not os.path.isfile(ply_file):
            raise FileNotFoundError(f"找不到点云文件：{ply_file}")
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points, dtype=np.float64)

        # --- 2. 读取相机轨迹 ---
        traj_file = os.path.join(self.output_dir, "optimized_camera_trajectory.txt")
        if not os.path.isfile(traj_file):
            raise FileNotFoundError(f"找不到相机轨迹文件：{traj_file}")
        poses = []
        with open(traj_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                # 格式：CamIdx tx ty tz qx qy qz qw
                _, tx, ty, tz, qx, qy, qz, qw = parts
                tx, ty, tz = map(float, (tx, ty, tz))
                qx, qy, qz, qw = map(float, (qx, qy, qz, qw))
                R_mat = R_scipy.from_quat([qx, qy, qz, qw]).as_matrix()
                t_vec = np.array([tx, ty, tz]).reshape(3, 1)
                poses.append({'R': R_mat, 't': t_vec})

        # --- 3. 可视化 ---
        print(f"[INFO] Loaded {len(points)} points and {len(poses)} camera poses from '{self.output_dir}'")
        visualize_with_open3d(points,poses)


  
if __name__ == "__main__":
    # Set your image folder path here
    image_folder = 'images'
    output_dir = 'output'
    camera_intrinsics = 'camera_intrinsic.txt'  # Path to your camera intrinsics file

    pipeline = ReconstructionPipeline(
        image_folder=image_folder,
        output_dir=output_dir,
        camera_intrinsics=camera_intrinsics,
        feature_extractor=True,
        feature_matcher=True,
        initial_pose_estimator=True,
        pnp_estimator=True,
        bundle_adjustment=True,
        verbose= False,
        save=False,
        load=True
    )
    pipeline.run()
    pipeline.visualize_from_output()  # 可视化输出结果

