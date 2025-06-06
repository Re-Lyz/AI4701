"""
Bundle Adjustment Module - 场景优化模块
功能：同时优化相机位姿和三维点坐标，最小化重投影误差
输出：优化后的点云和相机位姿
"""
import numpy as np
import scipy.optimize as opt
import os
import cv2
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import open3d as o3d

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BundleAdjustment:
    """Bundle Adjustment优化器类"""
    
    def __init__(self, camera_intrinsics, verbose=True):
        """
        初始化Bundle Adjustment优化器
        
        Args:
            camera_intrinsics: 相机内参矩阵 [3x3]
            verbose: 是否输出详细信息
        """
        self.K = camera_intrinsics
        self.verbose = verbose
        
    def rodrigues_to_rotation_matrix(self, rvec):
        """将旋转向量转换为旋转矩阵"""
        return cv2.Rodrigues(rvec)[0]
    
    def rotation_matrix_to_rodrigues(self, R):
        """将旋转矩阵转换为旋转向量"""
        return cv2.Rodrigues(R)[0].flatten()
    
    def project_points(self, points_3d, rvec, tvec, K):
        """
        将3D点投影到2D图像平面
        
        Args:
            points_3d: 3D点坐标 [Nx3]
            rvec: 旋转向量 [3]
            tvec: 平移向量 [3]
            K: 相机内参矩阵 [3x3]
            
        Returns:
            points_2d: 投影后的2D点 [Nx2]
        """
        # 使用OpenCV的投影函数
        points_2d, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3), 
            rvec, 
            tvec, 
            K, 
            None
        )
        return points_2d.reshape(-1, 2)
    
    def reprojection_error(self, params, observations, point_indices, camera_indices, n_cameras, n_points):
        """
        计算重投影误差
        
        Args:
            params: 优化参数 [相机参数 + 3D点坐标]
            observations: 观测的2D点坐标
            point_indices: 点索引
            camera_indices: 相机索引
            n_cameras: 相机数量
            n_points: 3D点数量
            
        Returns:
            residuals: 残差向量
        """
        # 解析参数
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        
        residuals = []
        
        for i, (point_idx, camera_idx) in enumerate(zip(point_indices, camera_indices)):
            # 获取相机参数
            rvec = camera_params[camera_idx, :3]
            tvec = camera_params[camera_idx, 3:6]
            
            # 获取3D点
            point_3d = points_3d[point_idx]
            
            # 投影到2D
            projected = self.project_points(
                point_3d.reshape(1, 3), rvec, tvec, self.K
            )[0]
            
            # 计算残差
            observed = observations[i]
            residual = projected - observed
            residuals.extend(residual)
        
        return np.array(residuals)
    
    def optimize(self, cameras, points_3d, observations, point_indices, camera_indices):
        """
        执行Bundle Adjustment优化
        
        Args:
            cameras: 初始相机位姿 [Nx4x4] (变换矩阵)
            points_3d: 初始3D点坐标 [Mx3]
            observations: 观测的2D点坐标 [Kx2]
            point_indices: 每个观测对应的3D点索引 [K]
            camera_indices: 每个观测对应的相机索引 [K]
            
        Returns:
            optimized_cameras: 优化后的相机位姿
            optimized_points_3d: 优化后的3D点坐标
            optimization_info: 优化信息
        """
        n_cameras = len(cameras)
        n_points = len(points_3d)
        n_observations = len(observations)
        
        logger.info(f"开始Bundle Adjustment优化:")
        logger.info(f"  相机数量: {n_cameras}")
        logger.info(f"  3D点数量: {n_points}")
        logger.info(f"  观测数量: {n_observations}")
        
        # 将相机位姿转换为参数向量
        camera_params = []
        for camera in cameras:
            if camera.shape == (4, 4):
                # 变换矩阵格式
                R_mat = camera[:3, :3]
                t_vec = camera[:3, 3]
            elif camera.shape == (3, 4):
                # [R|t]格式
                R_mat = camera[:, :3]
                t_vec = camera[:, 3]
            else:
                raise ValueError(f"不支持的相机位姿格式: {camera.shape}")
            
            # 转换为旋转向量
            rvec = self.rotation_matrix_to_rodrigues(R_mat)
            camera_params.extend(np.concatenate([rvec, t_vec]))
        
        # 构建初始参数向量
        initial_params = np.concatenate([
            np.array(camera_params),
            points_3d.ravel()
        ])
        
        logger.info(f"参数向量长度: {len(initial_params)}")
        
        # 执行非线性最小二乘优化
        result = opt.least_squares(
            self.reprojection_error,
            initial_params,
            args=(observations, point_indices, camera_indices, n_cameras, n_points),
            verbose=2 if self.verbose else 0,
            max_nfev=1000,
            ftol=1e-8,
            xtol=1e-8
        )
        
        # 解析优化结果
        optimized_params = result.x
        camera_params_opt = optimized_params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d_opt = optimized_params[n_cameras * 6:].reshape((n_points, 3))
        
        # 转换相机参数回矩阵格式
        optimized_cameras = []
        for i in range(n_cameras):
            rvec = camera_params_opt[i, :3]
            tvec = camera_params_opt[i, 3:6]
            
            R_mat = self.rodrigues_to_rotation_matrix(rvec)
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = tvec
            optimized_cameras.append(T)
        
        # 计算优化统计信息
        final_error = np.mean(result.fun**2)
        initial_error = np.mean(self.reprojection_error(
            initial_params, observations, point_indices, camera_indices, n_cameras, n_points
        )**2)
        
        optimization_info = {
            'success': result.success,
            'initial_cost': initial_error,
            'final_cost': final_error,
            'cost_reduction': initial_error - final_error,
            'iterations': result.nfev,
            'message': result.message
        }
        
        logger.info(f"优化完成:")
        logger.info(f"  成功: {result.success}")
        logger.info(f"  初始误差: {initial_error:.6f}")
        logger.info(f"  最终误差: {final_error:.6f}")
        logger.info(f"  误差降低: {initial_error - final_error:.6f}")
        logger.info(f"  迭代次数: {result.nfev}")
        
        return np.array(optimized_cameras), points_3d_opt, optimization_info

class BundleAdjustmentPipeline:
    """Bundle Adjustment处理流水线"""
    
    def __init__(self, config=None):
        """
        初始化BA流水线
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
    def load_data(self, pnp_recon_folder):
        """
        从PnP重建结果中加载数据
        
        Args:
            pnp_recon_folder: PnP重建结果文件夹
            
        Returns:
            data: 包含相机位姿、3D点、2D观测等的字典
        """
        logger.info(f"从 {pnp_recon_folder} 加载数据...")
        
        data = {}
        
        # 加载相机内参
        intrinsics_path = os.path.join(pnp_recon_folder, 'camera_intrinsic.txt')
        if os.path.exists(intrinsics_path):
            data['K'] = np.loadtxt(intrinsics_path)
        else:
            # 使用默认内参
            data['K'] = np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ], dtype=np.float64)
            logger.warning("未找到相机内参文件，使用默认值")
        
        # 加载相机位姿
        poses_path = os.path.join(pnp_recon_folder, 'camera_poses.npy')
        if os.path.exists(poses_path):
            data['cameras'] = np.load(poses_path)
        else:
            raise FileNotFoundError(f"未找到相机位姿文件: {poses_path}")
        
        # 加载3D点
        points_3d_path = os.path.join(pnp_recon_folder, 'points_3d.npy')
        if os.path.exists(points_3d_path):
            data['points_3d'] = np.load(points_3d_path)
        else:
            raise FileNotFoundError(f"未找到3D点文件: {points_3d_path}")
        
        # 加载2D观测点
        points_2d_path = os.path.join(pnp_recon_folder, 'points_2d.npy')
        if os.path.exists(points_2d_path):
            data['observations'] = np.load(points_2d_path)
        else:
            raise FileNotFoundError(f"未找到2D观测点文件: {points_2d_path}")
        
        # 加载索引信息
        indices_path = os.path.join(pnp_recon_folder, 'observation_indices.npz')
        if os.path.exists(indices_path):
            indices = np.load(indices_path)
            data['point_indices'] = indices['point_indices']
            data['camera_indices'] = indices['camera_indices']
        else:
            # 如果没有索引文件，生成默认索引
            n_obs = len(data['observations'])
            n_points = len(data['points_3d'])
            n_cameras = len(data['cameras'])
            
            data['point_indices'] = np.random.randint(0, n_points, n_obs)
            data['camera_indices'] = np.random.randint(0, n_cameras, n_obs)
            logger.warning("未找到观测索引文件，生成随机索引")
        
        logger.info(f"数据加载完成:")
        logger.info(f"  相机数量: {len(data['cameras'])}")
        logger.info(f"  3D点数量: {len(data['points_3d'])}")
        logger.info(f"  观测数量: {len(data['observations'])}")
        
        return data
    
    def save_results(self, cameras, points_3d, optimization_info, output_folder):
        """
        保存优化结果
        
        Args:
            cameras: 优化后的相机位姿
            points_3d: 优化后的3D点
            optimization_info: 优化信息
            output_folder: 输出文件夹
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # 保存相机位姿
        np.save(os.path.join(output_folder, 'optimized_cameras.npy'), cameras)
        
        # 保存3D点云
        np.save(os.path.join(output_folder, 'optimized_points_3d.npy'), points_3d)
        
        # 保存点云为PLY格式
        self.save_point_cloud_ply(points_3d, os.path.join(output_folder, 'point_cloud.ply'))
        
        # 保存相机轨迹
        self.save_camera_trajectory(cameras, os.path.join(output_folder, 'camera_trajectory.txt'))
        
        # 保存优化信息
        info_path = os.path.join(output_folder, 'optimization_info.txt')
        with open(info_path, 'w') as f:
            f.write("Bundle Adjustment Optimization Results\n")
            f.write("=" * 40 + "\n")
            for key, value in optimization_info.items():
                f.write(f"{key}: {value}\n")
        
        # 生成可视化图像
        self.visualize_results(cameras, points_3d, output_folder)
        
        logger.info(f"结果保存至: {output_folder}")
    
    def save_point_cloud_ply(self, points_3d, filepath):
        """保存点云为PLY格式"""
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for point in points_3d:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    def save_camera_trajectory(self, cameras, filepath):
        """保存相机轨迹"""
        with open(filepath, 'w') as f:
            f.write("# Camera Trajectory\n")
            f.write("# Format: timestamp tx ty tz qx qy qz qw\n")
            
            for i, camera in enumerate(cameras):
                # 提取位置
                position = camera[:3, 3]
                
                # 提取旋转并转换为四元数
                rotation_matrix = camera[:3, :3]
                rotation = R_scipy.from_matrix(rotation_matrix)
                quaternion = rotation.as_quat()  # [x, y, z, w]
                
                f.write(f"{i} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                       f"{quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")
    
    def visualize_results(self, cameras, points_3d, output_folder):
        """生成可视化结果"""
        fig = plt.figure(figsize=(15, 5))
        
        # 3D点云可视化
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                   c='blue', s=1, alpha=0.6)
        ax1.set_title('优化后的3D点云')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 相机轨迹可视化
        ax2 = fig.add_subplot(132, projection='3d')
        camera_positions = np.array([cam[:3, 3] for cam in cameras])
        ax2.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                'r-o', markersize=3)
        ax2.set_title('相机轨迹')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 整体场景可视化
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                   c='blue', s=1, alpha=0.3, label='3D Points')
        ax3.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                'r-o', markersize=3, label='Camera Trajectory')
        ax3.set_title('场景重建结果')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'reconstruction_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    from feature_extraction import extract_features_from_images
    from feature_matching import match_image_pairs,match_sift_features,filter_matches_by_homography
    from initial_recon import estimate_pose, print_pose_info
    from pnp_recon import pnp_pose, triangulate_initial_points,visualize_with_open3d
    
    # Test image paths
    image_paths = [
        "images/DJI_20200223_163016_842.jpg",
        "images/DJI_20200223_163017_967.jpg",
        "images/DJI_20200223_163018_942.jpg",
        "images/DJI_20200223_163019_752.jpg",
        "images/DJI_20200223_163020_712.jpg",
        "images/DJI_20200223_163021_627.jpg",
        "images/DJI_20200223_163022_557.jpg",
        "images/DJI_20200223_163023_427.jpg", 
    ]
    
    # Check if test images exist
    test_images_exist = all(os.path.exists(path) for path in image_paths)
    
    if not test_images_exist:
        print("Test images not found. Creating dummy test images...")
        # Create dummy test images with overlapping features
        for i, path in enumerate(image_paths):
            # Create images with some common features for matching
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some common geometric patterns
            cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
            cv2.circle(img, (300, 200), 40, (0, 0, 255), -1)
            cv2.line(img, (50, 300), (550, 300), (0, 255, 0), 3)
            
            # Add some unique features per image
            cv2.rectangle(img, (50+i*100, 50), (100+i*100, 100), (255, 0, 255), -1)
            cv2.circle(img, (400+i*50, 350), 25, (255, 255, 0), -1)
            
            cv2.imwrite(path, img)
        print("Test images created successfully!")
        
    print("\n" + "="*60)
    print("EXTRACTING FEATURES FOR MATCHING TEST")
    print("="*60)
    
    sift_features = extract_features_from_images(image_paths, method='sift')

    print("\n" + "="*60)
    print("MATCHING FEATURES BETWEEN TEST IMAGES")
    print("="*60)
    
    sift_matches = match_image_pairs(sift_features, method='sift')    
    
    if len(sift_features) >= 2:
        kp1, desc1 = sift_features[0]
        kp2, desc2 = sift_features[1]
        
        if desc1 is not None and desc2 is not None:
            print(f"\n--- Detailed analysis of pair (1, 2) ---")
            matches_1_2 = match_sift_features(desc1, desc2)
            
            # Apply homography filtering
            if len(matches_1_2) >= 4:
                filtered_matches_01, H01 = filter_matches_by_homography(kp1, kp2, matches_1_2)
                print(f"Filtered matches after homography: {len(filtered_matches_01)}")
                K_txt_path = "camera_intrinsic.txt"
                if not os.path.exists(K_txt_path):
                    raise FileNotFoundError(f"找不到相机内参文件：{K_txt_path}")
                K = np.loadtxt(K_txt_path, dtype=np.float32)
                if K.shape != (3, 3):
                    raise ValueError(f"读取到的相机内参矩阵形状不正确，应为 3x3，但得到 {K.shape}")
                
                # 估计相对位姿
                R, t, mask_pose, E, inlier_matches = estimate_pose(filtered_matches_01, kp1, kp2, K)
                
                # 打印位姿信息
                num_matches = len(filtered_matches_01)
                num_inliers = len(inlier_matches)
                print_pose_info(R, t, num_matches, num_inliers)
    R01, t01, mask_pose, E01, inlier_matches_01 = estimate_pose(filtered_matches_01, kp1, kp2, K)
    num_matches = len(filtered_matches_01)
    num_inliers = len(inlier_matches_01)
    print_pose_info(R01, t01, num_matches, num_inliers)

    # 6. 三角化得到初始三维点
    print("\nTriangulating initial 3D points...")
    initial_3d_points, valid_indices = triangulate_initial_points(inlier_matches_01, kp1, kp2, K, R01, t01)
    print(f"Triangulated {len(initial_3d_points)} initial 3D points")

    # 7. 构建初始的 camera_poses 列表 & point_observations 字典
    camera_poses = [
        {'R': np.eye(3),    't': np.zeros((3,1)),    'image': os.path.basename(image_paths[0])},
        {'R': R01,          't': t01,                'image': os.path.basename(image_paths[1])}
    ]
    scene_points_3d = initial_3d_points.copy()
    point_observations = {}
    # 对每个三角化的点，记录它在相机 0/1 中的像素观测
    for idx_pt3d, idx_match in enumerate(valid_indices):
        # 通过 idx_match 找到对应的 match 对象
        m = inlier_matches_01[idx_match]
        pt0 = kp1[m.queryIdx].pt  # 相机 0 的像素坐标
        pt1 = kp2[m.trainIdx].pt  # 相机 1 的像素坐标
        point_observations[idx_pt3d] = {0: pt0, 1: pt1}

    # 8. 递增式地对后续图像使用 PnP 恢复相机位姿
    num_images = 3   # 假设我们有 3 张图：0,1,2
    for img_idx in range(2, num_images):
        img_name = os.path.basename(image_paths[img_idx])
        print(f"\nProcessing PnP for image {img_idx}: {img_name}")

        # 提取第 img_idx 张图的特征（假设你已经提前生成过，或者在此处直接调用）
        # 这里用 extract_features_from_images 对 image_paths[img_idx] 单独提取
        kp_new, desc_new = extract_features_from_images([image_paths[img_idx]], method="sift")[0]

        # 0 ↔ img_idx 的匹配（假设你可以直接调用 match_sift_features）
        matches_0_i = match_sift_features(desc1, desc_new)  # desc1=第0张图的描述子
        if len(matches_0_i) < 6:
            print(f"  - 匹配点太少（{len(matches_0_i)}），跳过此张图")
            continue
        
        # 从 matches_0_i 中筛选那些在 point_observations 中已有三维点的对应
        image_points = []
        object_points = []
        for m in matches_0_i:
            idx0 = m.queryIdx   # 第0张图中 keypoint 索引
            idx_new = m.trainIdx  # 新图中 keypoint 索引
            
            # 遍历 point_observations，看第0张图的第 idx0 个 keypoint 是否属于某个三维点
            for pt3d_idx, obs_dict in point_observations.items():
                if 0 in obs_dict:
                    if np.linalg.norm(np.array(obs_dict[0]) - np.array(kp1[idx0].pt)) < 1.5:
                        # 找到了一个已有的三维点，可以作为 2D→3D 对应
                        image_points.append(kp_new[idx_new].pt)
                        object_points.append(scene_points_3d[pt3d_idx])
                        break
        
        print(f"  - 找到 {len(object_points)} 对2D-3D 对应，用于 PnP")
        if len(object_points) < 6:
            print("  - 2D-3D 对应不足 6，跳过 PnP")
            continue
        
        # 调用 PnP 求解新相机位姿
        try:
            R_new, t_new, inliers_new = pnp_pose(image_points, object_points, K)
            print(f"  - PnP 成功：{len(inliers_new)} 个内点 / 共 {len(object_points)} 对对应")
            camera_poses.append({'R': R_new, 't': t_new, 'image': img_name})
        except Exception as e:
            print(f"  - PnP 求解失败：{e}")
            continue
    # -----------------------
    # ——— 2. 准备 BA 数据并调用优化 ———
    # -----------------------

    # 2.1 构建 BA 所需的观测列表：observations, point_indices, camera_indices
    #     我们对每个三维点，查它在哪些相机里有观测，把每条观测当成一条记录
    obs_list = []        # 存放所有 2D 观测点 (u,v)
    pt_idx_list = []     # 对应的三维点索引 idx
    cam_idx_list = []    # 对应的相机索引 idx

    n_cams = len(camera_poses)
    n_pts  = scene_points_3d.shape[0]

    # 遍历每一个三维点 (pt3d_idx)，它的观测在 point_observations[pt3d_idx] 里
    for pt3d_idx, cam_dict in point_observations.items():
        for cam_idx, uv in cam_dict.items():
            # cam_idx 一定 < n_cams
            if cam_idx >= n_cams:
                continue
            obs_list.append([uv[0], uv[1]])
            pt_idx_list.append(pt3d_idx)
            cam_idx_list.append(cam_idx)

    observations   = np.array(obs_list, dtype=np.float64)       # K×2
    point_indices  = np.array(pt_idx_list, dtype=np.int32)      # 长度 K
    camera_indices = np.array(cam_idx_list, dtype=np.int32)     # 长度 K

    print("\n=== Bundle Adjustment 输入信息 ===")
    print(f"  相机数: {n_cams}")
    print(f"  三维点数: {n_pts}")
    print(f"  观测数: {observations.shape[0]}")

    # 2.2 构造相机姿态矩阵列表：4×4 矩阵
    cam_transforms = []
    for cam in camera_poses:
        R_mat = cam['R']
        t_vec = cam['t'].reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_mat
        T[:3, 3] = t_vec
        cam_transforms.append(T)

    # 2.3 调用 Bundle Adjustment 优化
    ba = BundleAdjustment(camera_intrinsics=K, verbose=True)
    optimized_cams, optimized_pts3d, ba_info = ba.optimize(
        cameras=cam_transforms,
        points_3d=scene_points_3d,
        observations=observations,
        point_indices=point_indices,
        camera_indices=camera_indices
    )

    # -----------------------
    # ——— 3. 保存 & 可视化 BA 结果 ———
    # -----------------------

    output_folder = "ba_results"
    os.makedirs(output_folder, exist_ok=True)

    # 3.1 把优化后相机矩阵保存为 NumPy
    np.save(os.path.join(output_folder, "optimized_cameras.npy"), np.stack(optimized_cams))

    # 3.2 把优化后三维点保存为 NumPy
    np.save(os.path.join(output_folder, "optimized_points_3d.npy"), optimized_pts3d)

    # 3.3 保存点云 PLY
    ply_path = os.path.join(output_folder, "optimized_point_cloud.ply")
    with open(ply_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {optimized_pts3d.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for pt in optimized_pts3d:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
    print(f"[INFO] Saved optimized point cloud to: {ply_path}")

    # 3.4 保存相机轨迹 txt
    traj_path = os.path.join(output_folder, "optimized_camera_trajectory.txt")
    with open(traj_path, 'w') as f:
        f.write("# CamIdx tx ty tz qx qy qz qw\n")
        for i, T in enumerate(optimized_cams):
            pos = T[:3, 3]
            R_mat = T[:3, :3]
            quat = R.from_matrix(R_mat).as_quat()  # (x,y,z,w)
            f.write(f"{i} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
    print(f"[INFO] Saved optimized camera trajectory to: {traj_path}")

    # 3.5 可视化优化后结果：使用 Open3D 弹交互窗口
    #      相机坐标系和点云同时画出
    pcd_opt = o3d.geometry.PointCloud()
    pcd_opt.points = o3d.utility.Vector3dVector(optimized_pts3d)
    pcd_opt.paint_uniform_color([0.2, 0.7, 0.3])  # 绿色点云

    geometry_list = [pcd_opt]
    for T in optimized_cams:
        # 画每台相机坐标系
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        mesh_frame.transform(T)
        geometry_list.append(mesh_frame)

        # 画相机中心一个小球
        center = T[:3, 3]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        sphere.translate(center)
        geometry_list.append(sphere)

    print("\n[INFO] Launching Open3D window to view optimized scene ...")
    o3d.visualization.draw_geometries(geometry_list,
                                      window_name="BA Optimized Reconstruction",
                                      width=1280, height=720)

    # 3.6 同时生成一张静态可视化（Matplotlib 3D）
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(optimized_pts3d[:,0], optimized_pts3d[:,1], optimized_pts3d[:,2],
               c='gray', s=1, alpha=0.6, label='Optimized Points')
    cam_positions = np.array([T[:3,3] for T in optimized_cams])
    ax.plot(cam_positions[:,0], cam_positions[:,1], cam_positions[:,2],
            'r-o', label='Optimized Camera Traj')
    ax.set_title("BA 优化后场景")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "ba_visualization.png"), dpi=300)
    plt.close()
    print(f"[INFO] Saved Matplotlib visualization to: {os.path.join(output_folder, 'ba_visualization.png')}")

    # 3.7 输出 BA 优化信息
    info_path = os.path.join(output_folder, "ba_optimization_info.txt")
    with open(info_path, 'w') as f:
        f.write("Bundle Adjustment Optimization Info\n")
        f.write("="*40 + "\n")
        for k,v in ba_info.items():
            f.write(f"{k}: {v}\n")
    print(f"[INFO] Saved BA optimization info to: {info_path}")

    print("\n=== 全流程已完成: PnP 重建 + BA 优化 ===")
    


