import cv2
import numpy as np
import os
import open3d as o3d
from typing import List, Tuple,Dict


def pnp_pose(image_points, object_points, K):
    """
    Estimate camera pose using PnP with RANSAC
    """
    object_points = np.array(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.array(image_points, dtype=np.float32).reshape(-1, 2)

    if len(object_points) < 4:
        raise ValueError("Need at least 4 point correspondences for PnP")

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, K, None,
        iterationsCount=1000,
        reprojectionError=2.0,
        confidence=0.99
    )
    if not success:
        raise RuntimeError("PnP solver failed")

    R_mat, _ = cv2.Rodrigues(rvec)
    return R_mat, tvec, inliers

def triangulate_points(
    matches: List[cv2.DMatch],
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    check_cheirality: bool = False,
    reproj_threshold: float = 0.0
) -> Tuple[np.ndarray, List[int]]:
    # 相机矩阵
    P1 = K @ np.hstack((R1, t1.reshape(3,1)))
    P2 = K @ np.hstack((R2, t2.reshape(3,1)))

    # 批量构造 2×N pts1, pts2
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).T
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).T

    # 三角化，得到 4×N 齐次坐标
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts4d /= pts4d[3:4,:]  # 齐次归一
    pts3d_all = pts4d[:3,:].T  # N×3

    # 齐次坐标投影回相机 1 和相机 2
    if check_cheirality or reproj_threshold > 0:
        # 批量构造齐次坐标
        Xh = np.vstack((pts3d_all.T, np.ones((1, pts3d_all.shape[0]))))  # 4×N
        # 投影
        proj1 = P1 @ Xh  # 3×N
        proj2 = P2 @ Xh
        # 归一化
        p1 = (proj1[:2] / proj1[2:3]).T  # N×2
        p2 = (proj2[:2] / proj2[2:3]).T

        # cheirality
        mask = np.ones((pts3d_all.shape[0],), dtype=bool)
        if check_cheirality:
            z1 = proj1[2]
            z2 = proj2[2]
            mask &= (z1 > 0) & (z2 > 0)

        # reprojection error
        if reproj_threshold > 0:
            err1 = np.linalg.norm(p1 - pts1.T, axis=1)
            err2 = np.linalg.norm(p2 - pts2.T, axis=1)
            mask &= (err1 <= reproj_threshold) & (err2 <= reproj_threshold)

        valid_idxs = np.nonzero(mask)[0].tolist()
        valid_pts = pts3d_all[mask]
    else:
        valid_idxs = list(range(pts3d_all.shape[0]))
        valid_pts = pts3d_all

    return valid_pts, valid_idxs

def multi_view_triangulation(
    new_idx: int,
    anchor_idxs: List[int],
    features: List[Tuple[List[cv2.KeyPoint], np.ndarray]],
    al_matches: List[List[List[cv2.DMatch]]],
    camera_poses: Dict[int, Dict[str, np.ndarray]],
    points_3d: Dict[int, np.ndarray],      # <–– 新增
    K: np.ndarray,
    min_parallax_deg: float = 1.0,
    reproj_thresh: float = 5.0
) -> List[Tuple[int, np.ndarray, Dict[int, float]]]:
    """
    对 new_idx 和 anchor_idxs 中每对做两视图三角，返回新增点列表：
      (pid, X3d, {cam: reproj_error, ...})

    points_3d 用来确定下一个可用 pid。
    """
    new_points = []
    # 从已有点 ID 计算下一个可用 pid
    pid_start = max(points_3d.keys(), default=-1) + 1

    for a in anchor_idxs:
        kp_a, _ = features[a]
        kp_n, _ = features[new_idx]
        matches_an = al_matches[a][new_idx]

        R_a, t_a = camera_poses[a]['R'], camera_poses[a]['t']
        R_n, t_n = camera_poses[new_idx]['R'], camera_poses[new_idx]['t']

        P1 = K @ np.hstack([R_a, t_a])
        P2 = K @ np.hstack([R_n, t_n])

        pts1 = np.float32([kp_a[m.queryIdx].pt for m in matches_an]).T
        pts2 = np.float32([kp_n[m.trainIdx].pt for m in matches_an]).T
        X4 = cv2.triangulatePoints(P1, P2, pts1, pts2)
        X3 = (X4[:3] / X4[3]).T

        for i, m in enumerate(matches_an):
            X = X3[i]

            # cheirality 检验
            if (R_a[2] @ (X - t_a.flatten())) <= 0 or \
               (R_n[2] @ (X - t_n.flatten())) <= 0:
                continue

            # 重投影误差
            x1_proj, _ = cv2.projectPoints(
                X.reshape(1,3), cv2.Rodrigues(R_a)[0], t_a, K, None
            )
            x2_proj, _ = cv2.projectPoints(
                X.reshape(1,3), cv2.Rodrigues(R_n)[0], t_n, K, None
            )
            err1 = np.linalg.norm(x1_proj.ravel() - kp_a[m.queryIdx].pt)
            err2 = np.linalg.norm(x2_proj.ravel() - kp_n[m.trainIdx].pt)
            if err1 > reproj_thresh or err2 > reproj_thresh:
                continue

            # 视差角度过滤
            ray1 = X - t_a.flatten()
            ray2 = X - t_n.flatten()
            angle = np.arccos(
                np.dot(ray1, ray2) /
                (np.linalg.norm(ray1) * np.linalg.norm(ray2))
            )
            if np.degrees(angle) < min_parallax_deg:
                continue
            match_map = {
                a: matches_an[i],        # matches_an 是 anchor→new 的 DMatch 列表
                new_idx: matches_an[i]
            }
            
            pid = pid_start
            pid_start += 1
            new_points.append((pid, X, match_map))

    return new_points

def visualize_with_open3d(
    scene_points_3d: np.ndarray,
    camera_mats: List[dict],      
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    show_cameras: bool = False
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Reconstructed Scene', width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color)
    opt.point_size = 1.0

    # 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points_3d)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(scene_points_3d))
    vis.add_geometry(pcd)

    if show_cameras:
        for pose in camera_mats:  # 修复：使用 camera_mats 而不是 camera_poses
            # 把 dict → 4×4 矩阵
            if isinstance(pose, dict):
                Rm = pose['R']
                tv = pose['t'].reshape(3,)
                T = np.eye(4, dtype=np.float64)   
                T[:3,:3] = Rm
                T[:3,3] = tv
            else:
                # 如果已经是4x4矩阵
                T = pose.astype(np.float64)

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            frame.transform(T)
            vis.add_geometry(frame)

    # —— 用第一个相机作为视角 —— 
    first_cam = camera_mats[0]
    
    # 处理不同的输入格式
    if isinstance(first_cam, dict):
        R0 = first_cam['R'].astype(np.float64)
        t0 = first_cam['t'].reshape(3).astype(np.float64)
        T0 = np.eye(4, dtype=np.float64)
        T0[:3, :3] = R0
        T0[:3, 3] = t0
    else:
        T0 = first_cam.astype(np.float64)
        R0 = T0[:3, :3]
        t0 = T0[:3, 3]

    # 相机中心（世界坐标系中的位置）
    cam_center = t0
    lookat = scene_points_3d.mean(axis=0)
    
    # 计算相机朝向
    # 在计算机视觉中，相机通常看向-Z方向
    front = (cam_center - lookat)
    front /= np.linalg.norm(front)

    # up 是相机的-Y方向（因为图像坐标系Y向下，世界坐标系Y向上）
    up_vec = R0 @ np.array([0, -1, 0], dtype=np.float64)  # 修复：使用-Y
    up = up_vec / np.linalg.norm(up_vec)

    # lookat 点设置在相机前方


    # 设置视角 - 需要在添加所有几何体后设置
    vis.poll_events()
    vis.update_renderer()
    
    ctr = vis.get_view_control()
    ctr.set_lookat(lookat.tolist())  # 转换为list
    ctr.set_front(front.tolist())
    ctr.set_up(up.tolist())
    ctr.set_zoom(0.1)

    vis.run()
    vis.destroy_window()

def visualize_colored_point_cloud(
    scene_points_3d: np.ndarray,
    camera_mats: np.ndarray,       # (N_cam,4,4)
    image_paths: List[str],        # 对应每台相机的 RGB 图路径列表
    K: np.ndarray,              # 相机内参矩阵 (3,3)
    color_cam_idx: int = 0,        # 用第几台相机的图像做取色
    bg_color=(1,1,1),
    zoom: float = 0.8
):
    """
    可视化带颜色的稀疏点云：
      - scene_points_3d: (N,3) 点云
      - camera_mats:      (N_cam,4,4) 外参齐次矩阵
      - image_paths:      N_cam 长度的 RGB 图路径
      - color_cam_idx:    选哪台相机的图像来给点着色
    """
    # 1) 读取要取色的那张 RGB 图
    color_img = cv2.imread(image_paths[color_cam_idx], cv2.IMREAD_COLOR)
    if color_img is None:
        raise ValueError(f"Cannot read color image: {image_paths[color_cam_idx]}")
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    h, w, _ = color_img.shape

    # 2) 投影点云到该相机
    T = camera_mats[color_cam_idx]
    R, t = T[:3,:3], T[:3,3]
    

    Xc = (R @ scene_points_3d.T) + t.reshape(3,1)   # (3,N)
    uv = K @ Xc                                      # (3,N)
    uv = uv[:2] / uv[2:3]                            # (2,N)
    uv = uv.T                                        # (N,2)

    # 3) 从图像中取色
    colors = np.zeros((scene_points_3d.shape[0], 3), dtype=np.float64)
    for i, (u, v) in enumerate(uv):
        ui = int(np.clip(round(u), 0, w-1))
        vi = int(np.clip(round(v), 0, h-1))
        colors[i] = color_img[vi, ui] / 255.0

    # 4) Open3D 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Colored Sparse Cloud", width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color)
    opt.point_size       = 2.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    # —— 用第一个相机作为视角 —— 
    first_cam = camera_mats[0]
    
    # 处理不同的输入格式
    if isinstance(first_cam, dict):
        R0 = first_cam['R'].astype(np.float64)
        t0 = first_cam['t'].reshape(3).astype(np.float64)
        T0 = np.eye(4, dtype=np.float64)
        T0[:3, :3] = R0
        T0[:3, 3] = t0
    else:
        T0 = first_cam.astype(np.float64)
        R0 = T0[:3, :3]
        t0 = T0[:3, 3]

    # 相机中心（世界坐标系中的位置）
    cam_center = t0
    lookat = scene_points_3d.mean(axis=0)
    
    # 计算相机朝向
    # 在计算机视觉中，相机通常看向-Z方向
    front = (cam_center - lookat)
    front /= np.linalg.norm(front)

    # up 是相机的-Y方向（因为图像坐标系Y向下，世界坐标系Y向上）
    up_vec = R0 @ np.array([0, -1, 0], dtype=np.float64)  # 修复：使用-Y
    up = up_vec / np.linalg.norm(up_vec)

    ctr = vis.get_view_control()
    ctr.set_lookat(lookat.tolist())
    ctr.set_front(front.tolist())
    ctr.set_up(up.tolist())
    ctr.set_zoom(zoom)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    from feature_extraction import extract_features_from_images
    from feature_matching import match_image_pairs, match_sift_features, filter_matches_by_homography
    from initial_recon import estimate_pose, print_pose_info
    
    # 图像路径列表
    image_paths = [
        "images/DJI_20200223_163016_842.jpg",
        "images/DJI_20200223_163017_967.jpg",
        "images/DJI_20200223_163018_942.jpg",
        "images/DJI_20200223_163019_752.jpg",
        "images/DJI_20200223_163020_712.jpg",
        "images/DJI_20200223_163021_627.jpg",
        "images/DJI_20200223_163022_557.jpg",
        "images/DJI_20200223_163023_427.jpg"
    ]

    # 确保测试图像存在
    test_images_exist = all(os.path.exists(path) for path in image_paths)
    if not test_images_exist:
        print("Test images not found. Creating dummy test images...")
        for i, path in enumerate(image_paths):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
            cv2.circle(img, (300, 200), 40, (0, 0, 255), -1)
            cv2.line(img, (50, 300), (550, 300), (0, 255, 0), 3)
            cv2.rectangle(img, (50 + i*100, 50), (100 + i*100, 100), (255, 0, 255), -1)
            cv2.circle(img, (400 + i*50, 350), 25, (255, 255, 0), -1)
            cv2.imwrite(path, img)
        print("Test images created successfully!")

    # 特征提取与匹配
    sift_features = extract_features_from_images(image_paths, method='sift')
    # 2. 两两匹配
    all_matches = match_image_pairs(sift_features, method='sift')

    # 3. 取 0↔1 并做 Homography 过滤
    kp1, desc1 = sift_features[0]
    kp2, desc2 = sift_features[1]
    raw_01 = all_matches[0][1]
    filtered_01, _ = filter_matches_by_homography(kp1, kp2, raw_01)

    # 4. 加载内参
    K = np.loadtxt("camera_intrinsic.txt", dtype=np.float32)

    # 5. 初始两视图位姿估计（基于本质矩阵）
    R01, t01, _, E01, inliers01 = estimate_pose(filtered_01, kp1, kp2, K)
    print_pose_info(R01, t01, len(filtered_01), len(inliers01))

    # 6. 三角化 3D 点
    scene_pts, valid_idxs = triangulate_points(filtered_01, kp1, kp2, K, np.eye(3), np.zeros((3,1)), R01, t01)
    camera_poses = [
        {'R':np.eye(3),'t':np.zeros((3,1)),'image':os.path.basename(image_paths[0])},
        {'R':R01,'t':t01,'image':os.path.basename(image_paths[1])}
    ]
    scene_points_3d = scene_pts.copy()
    point_obs = {idx:{0:sift_features[0][0][m.queryIdx].pt,1:sift_features[1][0][m.trainIdx].pt} 
                 for idx,m in enumerate(filtered_01) if idx in valid_idxs}

    # 7. 递增式 PnP
    for img_idx in range(2,len(image_paths)):
        kp_new, desc_new = sift_features[img_idx]
        raw_0i = all_matches[0][img_idx]
        filt_0i, _ = filter_matches_by_homography(kp1, kp_new, raw_0i)
        if len(filt_0i)<6: continue
        img_pts, obj_pts = [], []
        for m in filt_0i:
            for pid,obs in point_obs.items():
                if np.linalg.norm(np.array(obs[0])-np.array(sift_features[0][0][m.queryIdx].pt))<1.5:
                    img_pts.append(kp_new[m.trainIdx].pt);
                    obj_pts.append(scene_points_3d[pid]); break
        if len(obj_pts)<6: continue
        Rn, tn, ins = pnp_pose(img_pts, obj_pts, K)
        camera_poses.append({'R':Rn,'t':tn,'image':os.path.basename(image_paths[img_idx])})

    # 8. 可视化 & 总结打印
    visualize_with_open3d(scene_points_3d, camera_poses)
    print("\n"+"="*60); print("3D RECONSTRUCTION SUMMARY"); print("="*60)
    print(f"Total cameras: {len(camera_poses)} | Total points: {scene_points_3d.shape[0]}")
    for i,pose in enumerate(camera_poses):
        cw = -pose['R'].T @ pose['t'].flatten(); eu = np.degrees(cv2.Rodrigues(pose['R'])[0].flatten())
        print(f"\n Camera {i} ({pose['image']}): Pos={cw}, Euler={eu}")
