import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any,Tuple
from scipy.spatial.transform import Rotation as R_scipy
import open3d as o3d
from contextlib import redirect_stdout
from feature_extraction import load_or_extract_features
from feature_matching import load_or_match_image_pairs, filter_matches_by_homography, filter_two_view_geometry
from initial_recon import print_pose_info, pick_initial_pair
from pnp_recon import pnp_pose, visualize_with_open3d, triangulate_points
from bundle_adjustment import BundleAdjustment
from incremental_sfm import IncrementalSfM

def recenter_poses(poses: Dict[int,Tuple[np.ndarray,np.ndarray]], world_idx: int):
    R_w, t_w = poses[world_idx]
    new_poses = {}
    for k, (R_k, t_k) in poses.items():
        Rk_new = R_k @ R_w.T
        tk_new = t_k - Rk_new @ t_w
        new_poses[k] = (Rk_new, tk_new)
    return new_poses

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
        self._load_images()

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

    def _load_images(self):
        # 遍历文件夹，但先把 files 排序
        paths = []
        for root, dirs, files in os.walk(self.image_folder):
            files = sorted(files)  # 按文件名排序
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(root, f))
        self.image_paths = paths

    def get_world_idx(self, world_filename: str) -> int:
        """
        传入你想当世界坐标的文件名（不含路径），
        返回对应的 self.image_paths 索引。
        """
        # 方法一：利用 list.index，要求 image_paths 中的 path 是完整路径
        target = world_filename
        for idx, p in enumerate(self.image_paths):
            if os.path.basename(p) == target:
                return idx
        raise ValueError(f"无法在 image_paths 中找到文件 {world_filename}")        
        
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
                    inlier_matches, _ = _run_quiet(filter_two_view_geometry, kp1, kp2, matches, self.K)
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
            print(f"Initial pair: image {i} and image {j}")
            print_pose_info(R, t, len(inliers), len(inliers))
            # 准备 features 和 matches 的 dict 结构
            num_images = len(self.features)
            features_dict = {i: self.features[i] for i in range(num_images)}
            matches_dict = {}
            for i in range(num_images):
                for j in range(i+1, num_images):
                    m = self.al_matches[i][j]
                    if m:
                        matches_dict[(i, j)] = m
                    # 实例化并初始化
            sfm = IncrementalSfM(self.K, optimize_intrinsics=True)
            sfm.features = features_dict
            sfm.matches = matches_dict
            init_pair = (self.init_i, self.init_j)
            if not sfm.initialize_reconstruction(init_pair):
                raise RuntimeError("初始化失败，请检查初始图像对或匹配质量。")

        # 4. PnP Pose Estimation and Incremental Triangulation
        if self.pnp_estimator and len(self.poses) >= 2:

                    # 完整增量重建
            success = sfm.reconstruct_incremental(sfm.features, sfm.matches, init_pair)
            if not success:
                raise RuntimeError("增量重建中断。")
                    # 同步结果回原对象
            # 相机位姿
            self.poses = []
            self.registered_cams = list(sfm.registered_cams)
            for cam_idx, cam in sfm.cameras.items():
                if cam_idx in sfm.registered_cams:
                    self.poses.append({
                        'R': cam.R,
                        't': cam.t,
                        # 如果需要 inliers 信息，可以从 sfm.ransac 里缓存
                        'pair': None
                    })
                    # 3D 点坐标阵列
            self.points_3d = np.vstack([pt.xyz for pt in sfm.points_3d.values()])
                    # 点的观测字典
            self.point_observations = {
                pid: pt.observations
                for pid, pt in sfm.points_3d.items()
            }
            print(f"增量式重建完成：注册相机 {len(self.poses)}/{num_images}，三维点数 {len(self.points_3d)}")
            
        w_idx = self.get_world_idx("DJI_20200223_163016_842.jpg")
        print("world_idx =", w_idx)
        self.poses = recenter_poses(self.poses, world_idx=w_idx)

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
        bundle_adjustment=False,
        verbose= False,
        save=False,
        load=True
    )
    # pipeline.run()
    pipeline.visualize_from_output()  # 可视化输出结果

