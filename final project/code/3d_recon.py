import os
import sys
import time
import numpy as np
from typing import List, Dict, Any
from scipy.spatial.transform import Rotation as R_scipy
import open3d as o3d

from feature_extraction import extract_features_from_images
from feature_matching import match_image_pairs, match_sift_features, filter_matches_by_homography
from initial_recon import estimate_pose, print_pose_info
from pnp_recon import pnp_pose, triangulate_initial_points, visualize_with_open3d, triangulate_points, filter_matches_by_fundamental
from bundle_adjustment import BundleAdjustment

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
        bundle_adjustment: bool = True
    ):
        self.image_folder = image_folder
        self.camera_intrinsics = camera_intrinsics
        self.output_dir = output_dir
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.initial_pose_estimator = initial_pose_estimator
        self.pnp_estimator = pnp_estimator
        self.bundle_adjustment = bundle_adjustment

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

    def run(self):
        start_time = time.time()

        # 1. Feature Extraction
        if self.feature_extractor:
            print("Extracting features from images...")
            self.features = extract_features_from_images(self.image_paths, method='sift')

        # 2. Feature Matching
        if self.feature_matcher:
            print("Matching features between image pairs...")
            # 1. 特征提取后，直接拿到所有对的 matches
            all_matches = match_image_pairs(self.features, method='sift')
            
            # 2. 把满足阈值的那几对挑出来，构造 self.matches
            self.matches = []
            n = len(self.features)
            for i in range(n):
                for j in range(i+1, n):
                    matches_ij = all_matches[i][j]
                    if len(matches_ij) >= 4:
                        self.matches.append((i, j, matches_ij))

        # 3. Initial Pose Estimation
        if self.initial_pose_estimator and self.matches:
            print("Estimating initial pose from first image pair...")
            i, j, raw_matches = self.matches[0]
            kp1, _ = self.features[i]
            kp2, _ = self.features[j]
            filtered, H = filter_matches_by_homography(kp1, kp2, raw_matches)
            if len(filtered) < 4:
                raise RuntimeError("Homography inliers too few for initial pose estimation.")
            R, t, mask_pose, E, inliers = estimate_pose(filtered, kp1, kp2, self.K)
            self.poses = [
                {'R': np.eye(3), 't': np.zeros((3,1)), 'inliers': None, 'pair': (None, 0)},
                {'R': R, 't': t, 'inliers': inliers, 'pair': (i, j)}
            ]
            print_pose_info(R, t, len(filtered), len(inliers))

        # 4. PnP Pose Estimation and Triangulation
        if self.pnp_estimator and len(self.poses) >= 2:
            print("Triangulating initial 3D points...")
            first = self.poses[1]
            i, j = first['pair']
            kp1, _ = self.features[i]
            kp2, _ = self.features[j]
            initial_pts3d, valid_indices = triangulate_initial_points(
                first['inliers'], kp1, kp2, self.K, first['R'], first['t']
            )
            self.points_3d = initial_pts3d
            for idx_pt, match_idx in enumerate(valid_indices):
                m = first['inliers'][match_idx]
                pt0 = kp1[m.queryIdx].pt
                pt1 = kp2[m.trainIdx].pt
                self.point_observations[idx_pt] = {0: pt0, 1: pt1}

            for img_idx in range(2, len(self.image_paths)):
                print(f"\nProcessing PnP for image {img_idx}...")
                # 1) PnP 求解新相机外参
                kp_new, desc_new = self.features[img_idx]
                kp0, desc0 = self.features[0]
                matches_0_i = match_sift_features(desc0, desc_new)
                img_pts, obj_pts = [], []
                for m in matches_0_i:
                    # 遍历已有三维点的观测，找到 queryIdx 对应的 3D 点
                    for pt3d_idx, obs in self.point_observations.items():
                        if 0 in obs:
                            # obs[0] 是第0张图里该点的像素坐标
                            if np.linalg.norm(np.array(obs[0]) - np.array(kp0[m.queryIdx].pt)) < 1e-3:
                                # kp_new[m.trainIdx] 是新图里对应的像素坐标
                                img_pts.append(kp_new[m.trainIdx].pt)
                                # self.points_3d[pt3d_idx] 是这条 track 对应的 3D 坐标
                                obj_pts.append(self.points_3d[pt3d_idx])
                                break
                R_new, t_new, inliers_new = pnp_pose(img_pts, obj_pts, self.K)
                self.poses.append({'R': R_new, 't': t_new, 'inliers': inliers_new, 'pair': (0, img_idx)})
                print(f"  - PnP: {len(inliers_new)}/{len(obj_pts)} inliers")

                # 2) **增量三角化**：new_idx 与所有 prev_idx 做 matching → 基础矩阵滤外点 → 三角化
                for prev_idx in range(0, img_idx):
                    kp_prev, desc_prev = self.features[prev_idx]
                    # 2.1 匹配
                    matches_pi = match_sift_features(desc_prev, desc_new)
                    # 2.2 用基础矩阵 RANSAC 去除外点（你需要实现 filter_matches_by_fundamental）
                    inliers_pi, F = filter_matches_by_fundamental(kp_prev, kp_new, matches_pi)
                    if len(inliers_pi) < 8:
                        continue
                    # 2.3 三角化
                    R_prev, t_prev = self.poses[prev_idx]['R'], self.poses[prev_idx]['t']
                    new_pts3d, new_indices = triangulate_points(
                        inliers_pi, kp_prev, kp_new,
                        self.K, R_prev, t_prev, R_new, t_new
                    )
                    # 2.4 将新点累加进全局点云，更新自观测字典
                    base_idx = len(self.points_3d)
                    self.points_3d = np.vstack([self.points_3d, new_pts3d])
                    for local_i, pt_idx in enumerate(new_indices):
                        global_i = base_idx + local_i
                        m = inliers_pi[pt_idx]
                        pt_prev = kp_prev[m.queryIdx].pt
                        pt_new  = kp_new[m.trainIdx].pt
                        self.point_observations[global_i] = {
                            prev_idx: pt_prev,
                            img_idx:  pt_new
                        }
                                       
        camera_matrices_to_save = []
        for pose in self.poses:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = pose['R']
            T[:3,  3] = pose['t'].reshape(3,)
            camera_matrices_to_save.append(T)
            
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
        bundle_adjustment=False
    )
    pipeline.run()
    pipeline.visualize_from_output()  # 可视化输出结果

