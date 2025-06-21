# Scene Initialization using Epipolar Geometry
# Estimates relative pose between two cameras for 3D reconstruction initialization
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional

def pick_first_pair(
    matches: List[List[List[cv2.DMatch]]],
    features: List[Tuple[List[cv2.KeyPoint], np.ndarray]],
    K: np.ndarray
) -> Tuple[int, int, np.ndarray, np.ndarray, List[cv2.DMatch]]:
    """
    选取前两幅图（0 和 1）做初始化：
      - matches: 二维列表，matches[i][j] 存的是 i->j 的 match 列表
      - features: 每幅图的 (keypoints, descriptors)
      - K: 相机内参矩阵

    返回:
      init_i, init_j: 用作初始化的两张图的索引（这里固定是 0, 1）
      R0, t0: 相对位姿
      init_inliers: 被认为是 inlier 的 cv2.DMatch 列表
    """
    # 1. 指定索引
    init_i, init_j = 0, 5

    # 2. 取出它们的 matches 和 keypoints
    matches01 = matches[init_i][init_j]
    kp1, _ = features[init_i]
    kp2, _ = features[init_j]

    # 3. 调用 estimate_pose
    R0, t0, mask_pose, E, init_inliers = estimate_pose(matches01, kp1, kp2, K)

    # 4. 可选：打印信息
    num_matches = len(matches01)
    num_inliers = len(init_inliers)
    print_pose_info(R0, t0, num_matches, num_inliers)

    return init_i, init_j, R0, t0, init_inliers

def estimate_pose(matches, kp1, kp2, K):
    """
    Estimate relative pose between two cameras using essential matrix
    """
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Find essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover pose from essential matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    # Filter inlier matches
    inlier_matches = [matches[i] for i in range(len(matches)) if mask_pose[i]]
    
    return R, t, mask_pose, E, inlier_matches

def print_pose_info(R, t, num_matches, num_inliers):
    """
    Print detailed pose information
    """
    print("=" * 60)
    print("SCENE INITIALIZATION RESULTS")
    print("=" * 60)
    
    print(f"\n1. Feature Matching:")
    print(f"   - Total matches: {num_matches}")
    print(f"   - Inlier matches: {num_inliers}")
    print(f"   - Inlier ratio: {num_inliers/num_matches:.3f}")
    
    print(f"\n2. Rotation Matrix (R):")
    print(R)
    
    print(f"\n3. Translation Vector (t):")
    print(t.flatten())
    
    # Convert rotation matrix to Euler angles
    euler_angles = cv2.Rodrigues(R)[0].flatten()
    euler_degrees = np.degrees(euler_angles)
    
    print(f"\n4. Rotation as Rodrigues vector: {euler_angles}")
    print(f"5. Rotation in degrees: {euler_degrees}")
    
    # Calculate translation distance (normalized)
    translation_distance = np.linalg.norm(t)
    print(f"6. Translation distance (normalized): {translation_distance:.4f}")
    
    # Calculate rotation angle
    rotation_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"7. Total rotation angle: {rotation_angle:.2f} degrees")
    
    print("=" * 60)

def visualize_camera_poses(R, t, save_path=None):
    """
    Visualize the relative poses of two cameras
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera 1 (reference, at origin)
    cam1_pos = np.array([0, 0, 0])
    cam1_rot = np.eye(3)
    
    # Camera 2 (transformed)
    cam2_pos = -R.T @ t.flatten()  # Convert to world coordinates
    cam2_rot = R.T
    
    # Draw camera coordinate frames
    def draw_camera(ax, pos, rot, color, label, scale=0.5):
        # Camera position
        ax.scatter(*pos, color=color, s=150, marker='s', label=label)
        
        # Camera coordinate axes
        axes = rot * scale
        axis_colors = ['red', 'green', 'blue']
        axis_labels = ['X', 'Y', 'Z']
        
        for i, (axis_color, axis_label) in enumerate(zip(axis_colors, axis_labels)):
            end_pos = pos + axes[:, i]
            ax.plot([pos[0], end_pos[0]], [pos[1], end_pos[1]], [pos[2], end_pos[2]], 
                   color=axis_color, linewidth=3, alpha=0.8)
    
    # Draw cameras
    draw_camera(ax, cam1_pos, cam1_rot, 'blue', 'Camera 1 (Reference)')
    draw_camera(ax, cam2_pos, cam2_rot, 'red', 'Camera 2')
    
    # Draw connection between cameras (baseline)
    ax.plot([cam1_pos[0], cam2_pos[0]], [cam1_pos[1], cam2_pos[1]], [cam1_pos[2], cam2_pos[2]], 
           'k--', alpha=0.7, linewidth=2, label='Baseline')
    
    # Add camera viewing directions (optical axes)
    view_scale = 0.3
    cam1_view = cam1_pos + cam1_rot[:, 2] * view_scale  # Z-axis is optical axis
    cam2_view = cam2_pos + cam2_rot[:, 2] * view_scale
    
    ax.plot([cam1_pos[0], cam1_view[0]], [cam1_pos[1], cam1_view[1]], [cam1_pos[2], cam1_view[2]], 
           'b-', alpha=0.5, linewidth=2, label='Cam1 Optical Axis')
    ax.plot([cam2_pos[0], cam2_view[0]], [cam2_pos[1], cam2_view[1]], [cam2_pos[2], cam2_view[2]], 
           'r-', alpha=0.5, linewidth=2, label='Cam2 Optical Axis')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('Camera Pose Initialization\n(Relative Position and Orientation)')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = max(1.0, np.abs(cam2_pos).max()) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCamera pose visualization saved to: {save_path}")
    
    plt.show()

def visualize_epipolar_geometry(img1_path, img2_path, kp1, kp2, inlier_matches, K, save_path=None):
    """
    Visualize epipolar geometry with epilines (optional, if images are available)
    """
    try:
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            print("Images not found, skipping epipolar geometry visualization")
            return
            
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print("Could not load images for epipolar visualization")
            return
        
        # Select subset of matches for cleaner visualization
        step = max(1, len(inlier_matches) // 15)
        selected_matches = inlier_matches[::step]
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in selected_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in selected_matches])
        
        # Compute fundamental matrix
        F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        
        # Find epilines
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F).reshape(-1,3)
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F).reshape(-1,3)
        
        def draw_epilines(img, lines, pts):
            r, c = img.shape[:2]
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(lines)))[:, :3] * 255
            
            for i, (line, pt) in enumerate(zip(lines, pts)):
                color = tuple(map(int, colors[i]))
                x0, y0 = map(int, [0, -line[2]/line[1]]) if line[1] != 0 else (0, 0)
                x1, y1 = map(int, [c, -(line[2]+line[0]*c)/line[1]]) if line[1] != 0 else (c, 0)
                
                # Clip lines to image boundaries
                x0, y0 = max(0, min(c, x0)), max(0, min(r, y0))
                x1, y1 = max(0, min(c, x1)), max(0, min(r, y1))
                
                img_color = cv2.line(img_color, (x0,y0), (x1,y1), color, 1)
                img_color = cv2.circle(img_color, tuple(map(int, pt)), 4, color, -1)
            
            return img_color
        
        img1_epi = draw_epilines(img1, lines1, pts1)
        img2_epi = draw_epilines(img2, lines2, pts2)
        
        # Combine images
        combined = np.hstack([img1_epi, img2_epi])
        
        plt.figure(figsize=(15, 6))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title('Epipolar Geometry Visualization\n(Left: Image 1 with epilines from Image 2 | Right: Image 2 with epilines from Image 1)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Epipolar geometry visualization saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Could not create epipolar geometry visualization: {e}")

def pick_initial_pair(
    K: np.ndarray,
    al_matches: List[List[List[cv2.DMatch]]],
    keypoints: List[List[cv2.KeyPoint]],
    min_inliers: int = 15,
    min_triangulated: int = 50,
    min_tri_angle_deg: float = 1.5,
    reproj_error_thresh: float = 4.0,
    plane_ratio_thresh: float = 0.8,
    fixed_world_idx: Optional[int] = None
) -> Tuple[int, int, np.ndarray, np.ndarray, List[cv2.DMatch]]:
    """
    按 COLMAP 思路挑选初始化图像对：
      1) E vs H 退化检测
      2) recoverPose + cheirality & reprojection & 三角化角度过滤
      3) 排序: num_triangulated -> avg_angle -> n_inliers

    Args:
        K: 相机内参
        al_matches: N×N 匹配矩阵
        keypoints: 每张图的关键点列表
        min_inliers: 最小 recoverPose 内点
        min_triangulated: 最小可三角化稳定点数
        min_tri_angle_deg: 最小平均三角化角度 (°)
        reproj_error_thresh: 最大重投影误差 (px)
        plane_ratio_thresh: H/E 比例最大值，防止平面退化

    Returns:
        best_i, best_j, best_R, best_t, best_inliers
    """
    num_images = len(al_matches)
    best_score = (0, 0.0, 0)  # (num_tri, avg_angle, n_inliers)
    best_i = best_j = -1
    best_R = np.eye(3)
    best_t = np.zeros((3,1))
    best_inliers: List[cv2.DMatch] = []

    for i in range(num_images):
        for j in range(i+1, num_images):
            matches_ij = al_matches[i][j]
            if len(matches_ij) < min_inliers:
                continue

            pts1 = np.array([keypoints[i][m.queryIdx].pt for m in matches_ij])
            pts2 = np.array([keypoints[j][m.trainIdx].pt for m in matches_ij])

            # 1) 本质矩阵 & 单应矩阵 --> 平面退化检测
            E, mask_E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=1.0)
            if mask_E is None or mask_H is None:
                continue
            in_E = mask_E.ravel().astype(bool).sum()
            in_H = mask_H.ravel().astype(bool).sum()
            if in_H > plane_ratio_thresh * in_E:
                continue

            # 2) recoverPose
            _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask_E)
            pose_mask = mask_pose.ravel().astype(bool)
            n_inliers = pose_mask.sum()
            if n_inliers < min_inliers:
                continue

            inlier_pts1 = pts1[pose_mask]
            inlier_pts2 = pts2[pose_mask]

            # 3) 三角化验证: cheirality, reprojection & 三角化角度
            num_tri, avg_angle = compute_parallax(
                inlier_pts1, inlier_pts2, R, t, K, reproj_error_thresh
            )
            if num_tri < min_triangulated:
                continue
            if avg_angle < np.deg2rad(min_tri_angle_deg):
                continue

            # 4) 更新最优: 按 (num_tri, avg_angle, n_inliers)
            score = (num_tri, avg_angle, n_inliers)
            if score > best_score:
                best_score = score
                best_i, best_j = i, j
                best_R, best_t = R, t
                # 内点匹配按 pose_mask 过滤
                best_inliers = [m for m, ok in zip(matches_ij, pose_mask) if ok]

    if best_score[0] == 0:
        raise RuntimeError(
            "没有找到满足条件的初始图像对，请降低阈值或检查匹配质量。"
        )

    return best_i, best_j, best_R, best_t, best_inliers

def compute_parallax(
    pts1: np.ndarray,
    pts2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    reproj_error_thresh: float = 4.0
) -> Tuple[int, float]:
    """
    三角化 pts1-pts2, 并进行
      - 深度正性 (cheirality)
      - 重投影误差 (reproj_error_thresh px)
    统计满足条件的点数及平均视差角。

    Returns:
        num_valid, avg_angle (rad)
    """
    # 构建投影矩阵 P1, P2
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    # 三角化 (4×N)
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts4d /= pts4d[3:4]
    pts3d = pts4d[:3].T

    # 投影 & 误差
    proj1 = (P1 @ pts4d).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    proj2 = (P2 @ pts4d).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)

    # depth cheirality
    cam2_pts = (R @ pts3d.T + t).T
    valid = (pts3d[:,2] > 0) & (cam2_pts[:,2] > 0)
    valid &= (err1 < reproj_error_thresh) & (err2 < reproj_error_thresh)
    if not valid.any():
        return 0, 0.0

    # 视差角
    v1 = pts3d[valid] / np.linalg.norm(pts3d[valid], axis=1, keepdims=True)
    v2 = cam2_pts[valid] / np.linalg.norm(cam2_pts[valid], axis=1, keepdims=True)
    cosang = np.sum(v1 * v2, axis=1)
    cosang = np.clip(cosang, -1.0, 1.0)
    angles = np.arccos(cosang)
    return int(valid.sum()), float(np.mean(angles))


if __name__ == "__main__":
    from feature_extraction import extract_features_from_images
    from feature_matching import match_image_pairs,match_sift_features,filter_matches_by_homography
    
    # Test image paths
    image_paths = [
        "images/DJI_20200223_163016_842.jpg",
        "images/DJI_20200223_163017_967.jpg",
        "images/DJI_20200223_163018_942.jpg" 
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
                filtered_matches, homography = filter_matches_by_homography(kp1, kp2, matches_1_2)
                print(f"Filtered matches after homography: {len(filtered_matches)}")
                K_txt_path = "camera_intrinsic.txt"
                if not os.path.exists(K_txt_path):
                    raise FileNotFoundError(f"找不到相机内参文件：{K_txt_path}")
                K = np.loadtxt(K_txt_path, dtype=np.float32)
                if K.shape != (3, 3):
                    raise ValueError(f"读取到的相机内参矩阵形状不正确，应为 3x3，但得到 {K.shape}")
                
                # 估计相对位姿
                R, t, mask_pose, E, inlier_matches = estimate_pose(filtered_matches, kp1, kp2, K)
                
                # 打印位姿信息
                num_matches = len(filtered_matches)
                num_inliers = len(inlier_matches)
                print_pose_info(R, t, num_matches, num_inliers)
                
                # 可视化相机位姿并保存
                visualize_camera_poses(R, t, save_path="camera_poses.png")
                
                # 可视化极几何并保存
                visualize_epipolar_geometry(
                    image_paths[0],
                    image_paths[1],
                    kp1, kp2,
                    inlier_matches,
                    K,
                    save_path="epipolar_geometry.png"
                )    
            