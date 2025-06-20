import cv2
import numpy as np
import os
import open3d as o3d
from typing import List, Tuple


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

def visualize_with_open3d(
    scene_points_3d: np.ndarray,
    camera_poses: List[dict],
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    show_cameras: bool = False
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Reconstructed Scene', width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color)
    opt.point_size = 1.0

    # 点云：黑色前景
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points_3d)
    black = np.zeros((scene_points_3d.shape[0], 3))
    pcd.colors = o3d.utility.Vector3dVector(black)
    vis.add_geometry(pcd)

    if show_cameras:
        for pose in camera_poses:
            R_mat = pose['R']
            t_vec = pose['t'].reshape(3,)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3, origin=[0, 0, 0]
            )
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = t_vec
            mesh_frame.transform(T)
            vis.add_geometry(mesh_frame)

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
