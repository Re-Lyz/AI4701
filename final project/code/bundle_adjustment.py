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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BundleAdjustmentInterface:
    """
    Bundle Adjustment 基类接口，定义统一的构造与 optimize 方法
    """
    def __init__(self, camera_intrinsics: np.ndarray, verbose: bool = True):
        self.K = camera_intrinsics
        self.verbose = verbose

    def optimize(
        self,
        cameras: np.ndarray,
        points_3d: np.ndarray,
        observations: np.ndarray,
        point_indices: np.ndarray,
        camera_indices: np.ndarray
    ):
        """
        统一接口：
        Args:
            cameras: 初始相机位姿 np.ndarray [N_cameras×4×4]
            points_3d: 初始3D点坐标 np.ndarray [N_points×3]
            observations: 观测2D坐标 np.ndarray [N_obs×2]
            point_indices: 每个观测对应的3D点索引 np.ndarray [N_obs]
            camera_indices: 每个观测对应的相机索引 np.ndarray [N_obs]
        Returns:
            optimized_cameras: 优化后相机位姿 [N_cameras×4×4]
            optimized_points_3d: 优化后3D点 [N_points×3]
            info: dict 优化信息
        """
        raise NotImplementedError


class TorchBundleAdjustment(nn.Module):
    """基于 PyTorch 可微分的 Bundle Adjustment，带调试输出"""
    def __init__(
        self,
        camera_intrinsics: np.ndarray,
        init_cameras: np.ndarray,
        init_points: np.ndarray,
        verbose: bool = True
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.K_tensor = torch.from_numpy(camera_intrinsics).float().to(self.device)
        # 初始化 cams 和 points
        cams_param = []
        N_cam = init_cameras.shape[0]
        for i in range(N_cam):
            T = init_cameras[i]
            R = T[:3,:3]
            t = T[:3,3]
            rvec,_ = cv2.Rodrigues(R)
            cams_param.append(np.hstack([rvec.flatten(), t]))
        self.cams = nn.Parameter(torch.tensor(cams_param, dtype=torch.float32, device=self.device))
        self.points = nn.Parameter(torch.tensor(init_points, dtype=torch.float32, device=self.device))
        self.verbose = verbose

    def rodrigues(self, rvecs):
        if self.verbose:
            print(f"[rodrigues] input rvecs shape: {rvecs.shape}")
        theta = torch.norm(rvecs, dim=1, keepdim=True)
        axis = rvecs / (theta + 1e-8)
        cos_t = torch.cos(theta)[..., None]
        sin_t = torch.sin(theta)[..., None]
        K = torch.zeros((rvecs.shape[0],3,3), device=self.device)
        ax = axis
        K[:,0,1] = -ax[:,2]; K[:,0,2] = ax[:,1]
        K[:,1,0] = ax[:,2];  K[:,1,2] = -ax[:,0]
        K[:,2,0] = -ax[:,1]; K[:,2,1] = ax[:,0]
        I = torch.eye(3, device=self.device).unsqueeze(0)
        R = I*cos_t + (1-cos_t)*(axis.unsqueeze(-1)@axis.unsqueeze(-2)) + sin_t*K
        if self.verbose:
            print(f"[rodrigues] output R shape: {R.shape}")
        return R

    def forward(self, obs_uv, cam_idx, pt_idx):
        cams = self.cams[cam_idx]
        rvec = cams[:,:3]
        tvec = cams[:,3:6]
        R = self.rodrigues(rvec)
        pts = self.points[pt_idx]
        Xc = torch.bmm(R, pts.unsqueeze(-1)).squeeze(-1) + tvec
        uv = Xc / Xc[:,2:3]
        fx, fy = self.K_tensor[0,0], self.K_tensor[1,1]
        cx, cy = self.K_tensor[0,2], self.K_tensor[1,2]
        u = fx*uv[:,0] + cx
        v = fy*uv[:,1] + cy
        pred_uv = torch.stack([u,v], dim=1)
        if self.verbose:
            print(f"[forward] batch obs_uv shape: {obs_uv.shape}, pred_uv shape: {pred_uv.shape}")
            print(f"[forward] sample obs_uv: {obs_uv[:3].cpu().numpy()}, pred_uv: {pred_uv[:3].detach().cpu().numpy()}")
        return pred_uv

    def optimize(
        self,
        cameras: np.ndarray,
        points_3d: np.ndarray,
        observations: np.ndarray,
        point_indices: np.ndarray,
        camera_indices: np.ndarray,
        batch_size: int = 128
    ):
        # 准备数据
        obs = torch.tensor(observations, dtype=torch.float32, device=self.device)
        cam_idx = torch.tensor(camera_indices, dtype=torch.long, device=self.device)
        pt_idx  = torch.tensor(point_indices,  dtype=torch.long, device=self.device)
        dataset = TensorDataset(obs, cam_idx, pt_idx)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Adam 粗调
        adam = optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(10):
            epoch_loss = 0.0
            for batch_obs, batch_cam, batch_pt in loader:
                adam.zero_grad()
                pred_uv = self(batch_obs, batch_cam, batch_pt)
                loss    = (pred_uv - batch_obs).pow(2).sum()
                loss.backward()
                adam.step()
                epoch_loss += loss.item()
            if self.verbose:
                print(f"[optimize][Adam] epoch {epoch+1}/10, loss = {epoch_loss:.4f}")

        # LBFGS 精调
        if self.verbose:
            print("[optimize] Starting LBFGS refinement...")
        lbfgs = optim.LBFGS(self.parameters(), max_iter=50)
        def closure():
            lbfgs.zero_grad()
            pred_uv_full = self(obs, cam_idx, pt_idx)
            loss_full    = (pred_uv_full - obs).pow(2).sum()
            loss_full.backward()
            if self.verbose:
                print(f"[optimize][LBFGS] current loss = {loss_full.item():.4f}")
            return loss_full
        lbfgs.step(closure)

        # 导出结果
        cams_opt = self.cams.detach().cpu().numpy()
        pts_opt  = self.points.detach().cpu().numpy()
        optimized_cams = []
        for r_t in cams_opt:
            rvec, t = r_t[:3], r_t[3:6]
            R,_ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = t
            optimized_cams.append(T)

        if self.verbose:
            print(f"[optimize] Completed optimization. cams_opt shape: {cams_opt.shape}, points shape: {pts_opt.shape}")
        info = {'success': True, 'method': 'torch'}
        return np.stack(optimized_cams), pts_opt, info


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


if __name__ == "__main__":
    from feature_extraction import extract_features_from_images
    from feature_matching import match_sift_features, filter_two_view_geometry
    from initial_recon import estimate_pose, print_pose_info
    from pnp_recon import pnp_pose, triangulate_points

    image_folder = 'images'
    paths = []

    # 1. 遍历、收集
    for root, dirs, files in os.walk(image_folder):
        for f in sorted(files):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(root, f))

    # 2. 找到目标（不含后缀）  
    target_stem = 'DJI_20200223_163225_243'
    # 注意：如果有不同后缀（.jpg/.png），就只比对 stem
    target_path = next(
        (p for p in paths if os.path.splitext(os.path.basename(p))[0] == target_stem),
        None
    )

    # 3. 如果找到，就先移除再插入到最前面
    if target_path:
        paths.remove(target_path)
        paths.insert(0, target_path)

    image_paths = paths

    
    os.makedirs("images", exist_ok=True)
    for path in image_paths:
        if not os.path.exists(path):
            img = np.random.randint(0,255,(480,640,3),dtype=np.uint8)
            cv2.rectangle(img,(100,100),(200,200),(255,255,255),-1)
            cv2.circle(img,(300,200),40,(0,0,255),-1)
            cv2.line(img,(50,300),(550,300),(0,255,0),3)
            cv2.imwrite(path,img)

    # Load intrinsics
    K = np.loadtxt("camera_intrinsic.txt", dtype=np.float64)

    # 1. Extract and match features for first two
    kp1, desc1 = extract_features_from_images([image_paths[0]], method='sift')[0]
    kp2, desc2 = extract_features_from_images([image_paths[1]], method='sift')[0]
    matches_1_2 = match_sift_features(desc1, desc2)
    inliers_01, _ = filter_two_view_geometry(kp1, kp2, matches_1_2, K)
    R01, t01, _, _, inlier_matches_01 = estimate_pose(inliers_01, kp1, kp2, K)
    print_pose_info(R01, t01, len(inliers_01), len(inlier_matches_01))

    # 2. Triangulate initial points
    initial_3d_points, valid_indices = triangulate_points(
        inlier_matches_01, kp1, kp2, K,
        R1=np.eye(3), t1=np.zeros((3,1)), R2=R01, t2=t01
    )
    print(f"Triangulated {len(initial_3d_points)} points")

    # 3. Initialize poses and observations
    camera_poses = [
        {'R': np.eye(3), 't': np.zeros((3,1))},
        {'R': R01,       't': t01}
    ]
    scene_points_3d = initial_3d_points.tolist()
    point_observations = {}
    for local_idx, g_idx in enumerate(valid_indices):
        m = inlier_matches_01[g_idx]
        point_observations[local_idx] = {
            0: tuple(kp1[m.queryIdx].pt),
            1: tuple(kp2[m.trainIdx].pt)
        }

    # 4. Incremental PnP
    for img_idx in range(2, len(image_paths)):
        kp_new, desc_new = extract_features_from_images([image_paths[img_idx]], method='sift')[0]
        matches_0_i = match_sift_features(desc1, desc_new)
        image_pts, obj_pts, obj_ids = [], [], []
        for m in matches_0_i:
            for pid, obs in point_observations.items():
                if np.linalg.norm(np.array(obs[0]) - np.array(kp1[m.queryIdx].pt))<1.5:
                    image_pts.append(kp_new[m.trainIdx].pt)
                    obj_pts.append(scene_points_3d[pid])
                    obj_ids.append(pid)
                    break
        if len(obj_pts)<6:
            continue
        Rn, tn, inls = pnp_pose(image_pts, obj_pts, K)
        camera_poses.append({'R': Rn, 't': tn})
        inls = inls.flatten()  
        for idx in inls:
            pid = obj_ids[idx]
            point_observations[pid][img_idx] = tuple(image_pts[idx])

    # 5. Prepare BA inputs
    obs_list, pidx_list, cidx_list = [],[],[]
    for pid, obs in point_observations.items():
        for cid, uv in obs.items():
            obs_list.append(uv)
            pidx_list.append(pid)
            cidx_list.append(cid)
    observations   = np.array(obs_list)
    point_indices  = np.array(pidx_list)
    camera_indices = np.array(cidx_list)

    cam_transforms = []
    for cam in camera_poses:
        T = np.eye(4)
        T[:3,:3] = cam['R']
        T[:3,3]  = cam['t'].reshape(-1)
        cam_transforms.append(T)
    init_cams = np.stack(cam_transforms)
    pts_arr   = np.array(scene_points_3d)

    # 6. Bundle Adjustment
    ba = TorchBundleAdjustment(camera_intrinsics=K,
                                init_cameras=init_cams,
                                init_points = pts_arr)
    optimized_cams, optimized_pts3d, info = ba.optimize(
        cameras=init_cams,
        points_3d=pts_arr,
        observations=observations,
        point_indices=point_indices,
        camera_indices=camera_indices
    )

    print("BA Optimization Info:", info)

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
            quat = R_scipy.from_matrix(R_mat).as_quat()  # (x,y,z,w)
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
        for k,v in info.items():
            f.write(f"{k}: {v}\n")
    print(f"[INFO] Saved BA optimization info to: {info_path}")

    print("\n=== 全流程已完成: PnP 重建 + BA 优化 ===")
    


