# PnP-based 3D Scene Reconstruction
# Performs incremental 3D reconstruction using Perspective-n-Point (PnP) algorithm
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def triangulate_initial_points(matches, kp1, kp2, K, R, t):
    """
    Triangulate initial 3D points from the first two views
    """
    # Camera projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])  # First camera at origin
    P2 = K @ np.hstack([R, t.reshape(3, 1)])           # Second camera
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous coordinates
    
    # Filter points based on reprojection error and depth
    valid_points = []
    valid_indices = []
    
    for i, point_3d in enumerate(points_3d.T):
        # Check if point is in front of both cameras
        if point_3d[2] > 0 and (R @ point_3d + t.flatten())[2] > 0:
            # Check reprojection error
            point_3d_homo = np.append(point_3d, 1).reshape(4, 1)
            
            # Reproject to first camera
            proj1 = P1 @ point_3d_homo
            proj1 = proj1[:2] / proj1[2]
            error1 = np.linalg.norm(proj1.flatten() - pts1[:, i])
            
            # Reproject to second camera  
            proj2 = P2 @ point_3d_homo
            proj2 = proj2[:2] / proj2[2]
            error2 = np.linalg.norm(proj2.flatten() - pts2[:, i])
            
            # Accept point if reprojection error is small
            if error1 < 2.0 and error2 < 2.0:
                valid_points.append(point_3d)
                valid_indices.append(i)
    
    return np.array(valid_points), valid_indices

def pnp_pose(image_points, object_points, K):
    """
    Estimate camera pose using PnP with RANSAC
    """
    # Ensure correct data types
    object_points = np.array(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.array(image_points, dtype=np.float32).reshape(-1, 2)
    
    if len(object_points) < 4:
        raise ValueError("Need at least 4 point correspondences for PnP")
    
    # Solve PnP with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, K, None,
        iterationsCount=1000,
        reprojectionError=2.0,
        confidence=0.99
    )
    
    if not success:
        raise RuntimeError("PnP solver failed")
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    return R, tvec, inliers

def find_common_features(features1, features2, matches):
    """
    Find common features between two sets of keypoints using matches
    """
    common_indices1 = []
    common_indices2 = []
    
    for match in matches:
        common_indices1.append(match.queryIdx)
        common_indices2.append(match.trainIdx)
    
    return common_indices1, common_indices2

def incremental_reconstruction(initial_recon_folder, matches_folder, features_folder, K, num_images=3):
    """
    Perform incremental 3D reconstruction starting from initial pose estimation
    """
    print("=" * 60)
    print("INCREMENTAL 3D RECONSTRUCTION")
    print("=" * 60)
    
    # Load initial reconstruction results
    initial_data = np.load(os.path.join(initial_recon_folder, 'initial_pose.npz'), allow_pickle=True)
    R_init = initial_data['R']
    t_init = initial_data['t']  
    inlier_matches_01 = initial_data['inlier_matches']
    
    print(f"\nLoaded initial pose between cameras 0 and 1")
    print(f"Initial matches: {len(inlier_matches_01)}")
    
    # Load keypoints for first two images
    kp1 = np.load(os.path.join(features_folder, "0000.png_keypoints.npy"), allow_pickle=True)
    kp2 = np.load(os.path.join(features_folder, "0001.png_keypoints.npy"), allow_pickle=True)
    
    # Triangulate initial 3D points from first two views
    print("\nTriangulating initial 3D points...")
    initial_3d_points, valid_indices = triangulate_initial_points(inlier_matches_01, kp1, kp2, K, R_init, t_init)
    
    print(f"Triangulated {len(initial_3d_points)} initial 3D points")
    
    # Store camera poses and 3D points
    camera_poses = [
        {'R': np.eye(3), 't': np.zeros((3, 1)), 'image': '0000.png'},  # First camera at origin
        {'R': R_init, 't': t_init, 'image': '0001.png'}               # Second camera
    ]
    
    scene_points_3d = initial_3d_points.copy()
    point_observations = {}  # Track which cameras see which points
    
    # Initialize point observations for first two cameras
    for i, valid_idx in enumerate(valid_indices):
        match = inlier_matches_01[valid_idx]
        point_observations[i] = {
            0: kp1[match.queryIdx].pt,  # Camera 0 observation
            1: kp2[match.trainIdx].pt   # Camera 1 observation
        }
    
    # Add more cameras using PnP
    for img_idx in range(2, min(num_images, 10)):  # Limit to avoid too many images
        img_name = f"{img_idx:04d}.png"
        kp_file = os.path.join(features_folder, f"{img_name}_keypoints.npy")
        matches_file = os.path.join(matches_folder, f"matches_0_{img_idx}.npy")
        
        if not (os.path.exists(kp_file) and os.path.exists(matches_file)):
            print(f"Skipping image {img_name} - missing data files")
            continue
            
        print(f"\nProcessing image {img_name}...")
        
        # Load keypoints and matches
        kp_new = np.load(kp_file, allow_pickle=True)
        matches_0_new = np.load(matches_file, allow_pickle=True)
        
        if len(matches_0_new) < 10:
            print(f"Not enough matches ({len(matches_0_new)}) for image {img_name}")
            continue
        
        # Find 2D-3D correspondences
        image_points = []
        object_points = []
        
        for match in matches_0_new:
            kp0_idx = match.queryIdx
            kp_new_idx = match.trainIdx
            
            # Check if this keypoint from camera 0 corresponds to a 3D point
            for point_3d_idx, observations in point_observations.items():
                if 0 in observations:
                    # Check if the keypoint matches
                    obs_pt = observations[0]
                    curr_pt = kp1[kp0_idx].pt
                    
                    # Simple distance check (could be improved)
                    if np.linalg.norm(np.array(obs_pt) - np.array(curr_pt)) < 2.0:
                        image_points.append(kp_new[kp_new_idx].pt)
                        object_points.append(scene_points_3d[point_3d_idx])
                        break
        
        if len(object_points) < 6:
            print(f"Not enough 2D-3D correspondences ({len(object_points)}) for PnP")
            continue
        
        try:
            # Estimate pose using PnP
            R_new, t_new, inliers = pnp_pose(image_points, object_points, K)
            
            print(f"PnP successful - {len(inliers) if inliers is not None else 0} inliers from {len(object_points)} correspondences")
            
            # Add new camera to reconstruction
            camera_poses.append({
                'R': R_new, 
                't': t_new, 
                'image': img_name,
                'num_correspondences': len(object_points),
                'num_inliers': len(inliers) if inliers is not None else 0
            })
            
        except Exception as e:
            print(f"Failed to estimate pose for {img_name}: {e}")
            continue
    
    print(f"\nReconstructed {len(camera_poses)} camera poses")
    print(f"Total 3D points: {len(scene_points_3d)}")
    
    return camera_poses, scene_points_3d, point_observations

def visualize_reconstruction(camera_poses, scene_points_3d, save_path=None):
    """
    Visualize the 3D reconstruction result
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    if len(scene_points_3d) > 0:
        # Filter points for better visualization
        valid_points = scene_points_3d[np.abs(scene_points_3d).max(axis=1) < 20]
        if len(valid_points) > 0:
            ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
                      c='lightblue', alpha=0.6, s=1, label=f'3D Points ({len(valid_points)})')
    
    # Plot cameras
    colors = plt.cm.tab10(np.linspace(0, 1, len(camera_poses)))
    
    for i, (pose, color) in enumerate(zip(camera_poses, colors)):
        R = pose['R']
        t = pose['t'].flatten() if pose['t'].ndim > 1 else pose['t']
        
        # Camera position in world coordinates
        cam_pos = -R.T @ t
        
        # Camera orientation
        cam_rot = R.T
        
        # Plot camera position
        ax.scatter(*cam_pos, color=color, s=200, marker='s', 
                  label=f"Cam {i}: {pose['image']}")
        
        # Plot camera coordinate frame
        scale = 0.5
        axes = cam_rot * scale
        axis_colors = ['red', 'green', 'blue']
        
        for j, axis_color in enumerate(axis_colors):
            end_pos = cam_pos + axes[:, j]
            ax.plot([cam_pos[0], end_pos[0]], 
                   [cam_pos[1], end_pos[1]], 
                   [cam_pos[2], end_pos[2]], 
                   color=axis_color, linewidth=2, alpha=0.7)
        
        # Plot viewing direction
        view_dir = cam_pos + cam_rot[:, 2] * 0.8  # Z-axis is optical axis
        ax.plot([cam_pos[0], view_dir[0]], 
               [cam_pos[1], view_dir[1]], 
               [cam_pos[2], view_dir[2]], 
               color=color, linestyle='--', alpha=0.5, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scene Reconstruction Results\n(Cameras and Reconstructed Points)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set reasonable viewing limits
    if len(scene_points_3d) > 0:
        valid_points = scene_points_3d[np.abs(scene_points_3d).max(axis=1) < 20]
        if len(valid_points) > 0:
            center = np.mean(valid_points, axis=0)
            max_range = np.max(np.abs(valid_points - center)) * 1.2
        else:
            center = np.array([0, 0, 0])
            max_range = 5
    else:
        center = np.array([0, 0, 0])
        max_range = 5
    
    ax.set_xlim([center[0] - max_range, center[0] + max_range])
    ax.set_ylim([center[1] - max_range, center[1] + max_range])
    ax.set_zlim([center[2] - max_range, center[2] + max_range])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nReconstruction visualization saved to: {save_path}")
    
    plt.show()

def print_reconstruction_summary(camera_poses, scene_points_3d):
    """
    Print detailed reconstruction results
    """
    print("\n" + "=" * 60)
    print("3D RECONSTRUCTION SUMMARY")
    print("=" * 60)
    
    print(f"\n1. Reconstructed Cameras: {len(camera_poses)}")
    for i, pose in enumerate(camera_poses):
        R = pose['R']
        t = pose['t'].flatten() if pose['t'].ndim > 1 else pose['t']
        
        print(f"\n   Camera {i} ({pose['image']}):")
        print(f"   - Position: {-R.T @ t}")
        print(f"   - Rotation angles: {np.degrees(cv2.Rodrigues(R)[0].flatten())}")
        
        if 'num_correspondences' in pose:
            print(f"   - 2D-3D correspondences: {pose['num_correspondences']}")
            print(f"   - PnP inliers: {pose['num_inliers']}")
    
    print(f"\n2. Reconstructed 3D Points: {len(scene_points_3d)}")
    if len(scene_points_3d) > 0:
        print(f"   - X range: [{scene_points_3d[:, 0].min():.2f}, {scene_points_3d[:, 0].max():.2f}]")
        print(f"   - Y range: [{scene_points_3d[:, 1].min():.2f}, {scene_points_3d[:, 1].max():.2f}]")
        print(f"   - Z range: [{scene_points_3d[:, 2].min():.2f}, {scene_points_3d[:, 2].max():.2f}]")
    
    print("\n" + "=" * 60)

def visualize_camera_poses(R_list, t_list, save_path=None):
    """
    可视化所有相机（参考系在世界坐标系）和稀疏三维点
    （此处只是画相机坐标系，不包含点云；点云会在后面函数里单独可视化）
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0,1,len(R_list)))
    for i, (R, t, color) in enumerate(zip(R_list, t_list, colors)):
        # 世界坐标系下相机中心
        cam_pos = -R.T @ t.flatten()
        cam_rot = R.T
        
        ax.scatter(*cam_pos, color=color, s=100, marker='s', label=f"Cam {i}")
        
        # 画相机坐标系三轴（大小 scale=0.5）
        scale = 0.5
        axes = cam_rot * scale
        for axis_idx, axis_color in enumerate(['r','g','b']):
            end = cam_pos + axes[:,axis_idx]
            ax.plot([cam_pos[0], end[0]],
                    [cam_pos[1], end[1]],
                    [cam_pos[2], end[2]],
                    color=axis_color, linewidth=2)
        
        # 光轴（Z 轴）的延伸线
        view_dir = cam_pos + cam_rot[:,2] * (scale*1.2)
        ax.plot([cam_pos[0], view_dir[0]],
                [cam_pos[1], view_dir[1]],
                [cam_pos[2], view_dir[2]],
                color=color, linestyle='--', linewidth=1)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses in World Coordinate")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Camera poses visualization saved to: {save_path}")
    plt.show()

def visualize_scene_and_points(camera_poses, scene_points_3d, save_path=None):
    """
    同时可视化相机坐标系和场景三维点云
    camera_poses: list of {'R':…, 't':…, 'image':…}
    scene_points_3d: N x 3 ndarray
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制三维点（过滤掉过于离散的点，仅作示范）
    if scene_points_3d.size > 0:
        pts = scene_points_3d.copy()
        mask = np.all(np.abs(pts) < 20.0, axis=1)
        pts_vis = pts[mask]
        ax.scatter(pts_vis[:,0], pts_vis[:,1], pts_vis[:,2],
                   c='lightgray', alpha=0.6, s=2, label=f"3D Points ({len(pts_vis)})")
    
    # 绘制相机
    colors = plt.cm.tab10(np.linspace(0,1,len(camera_poses)))
    for i, (pose, color) in enumerate(zip(camera_poses, colors)):
        R = pose['R']
        t = pose['t'].flatten() if pose['t'].ndim>1 else pose['t']
        cam_pos = -R.T @ t
        cam_rot = R.T
        
        ax.scatter(*cam_pos, color=color, s=80, marker='^', label=f"Cam {i}: {pose['image']}")
        scale = 0.5
        axes = cam_rot * scale
        for axis_idx, axis_color in enumerate(['r','g','b']):
            end = cam_pos + axes[:,axis_idx]
            ax.plot([cam_pos[0], end[0]],
                    [cam_pos[1], end[1]],
                    [cam_pos[2], end[2]],
                    color=axis_color, linewidth=2)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scene Reconstruction (Cameras + Points)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scene reconstruction visualization saved to: {save_path}")
    plt.show()

def visualize_with_open3d(camera_poses, scene_points_3d):
    """
    使用 Open3D 可视化重建结果：
    - scene_points_3d: NumPy 数组，形状 (N,3)，N 个 3D 点
    - camera_poses: list of dict，每个 dict 包含：
        {
            'R': 3x3 旋转矩阵 (numpy.ndarray),
            't': 3x1 或长度为 3 的平移向量 (numpy.ndarray),
            'image': 相机对应的图像名字（可选，只用来标注）
        }
    """

    # 1. 将三维点构建成 Open3D 的 PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points_3d)
    # 也可以给点云加一点颜色（可选）
    # 例如：所有点都给成灰色
    colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (scene_points_3d.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 2. 构造摄像机坐标系（Coordinate Frame）和相机中心点
    camera_frames = []   # 用来存放每个相机的坐标系（open3d.geometry.TriangleMesh）
    camera_centers = []  # 用来存放每个相机的中心点（open3d.geometry.PointCloud）

    # 设置一个“相机坐标系”网格，用来表示相机的朝向与位置
    # 这里我们用 open3d.geometry.TriangleMesh.create_coordinate_frame() 制作一个坐标轴小模型
    for idx, pose in enumerate(camera_poses):
        R = pose['R']            # 3x3 ndarray
        t = pose['t'].reshape(3)  # 长度为 3 的向量

        # 每个相机都建立自己的坐标系 mesh（大小可以调整 scale 参数）
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3,       # 坐标轴的长度（可自行调整） 
            origin=[0, 0, 0]
        )

        # 对 mesh_frame 应用旋转和平移
        # Open3D 里需要先把旋转矩阵转换成 4x4 变换矩阵，再赋值
        # 注意：Open3D 使用右乘 4x4 坐标变换 T，这里 T 先把形状固定成 4x4
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R       # 左上 3×3 放 R
        T[:3, 3] = t        # 前 3 个元素是平移向量 t
        mesh_frame.transform(T)  # 对 mesh_frame 做变换，使坐标系移动到相机中心

        camera_frames.append(mesh_frame)

        # 同时画一个小球或点来表示相机中心
        cam_pt = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # 半径 0.05 的小球
        cam_pt.paint_uniform_color([1.0, 0.0, 0.0])  # 红色小球
        cam_pt.translate(t)  # 把小球移动到相机中心
        camera_centers.append(cam_pt)

    # 3. 把所有对象一起加入到一个 Open3D 场景里
    o3d.visualization.draw_geometries(
        [pcd] + camera_frames + camera_centers,
        window_name='Reconstructed Scene (with Open3D)',
        width=1280,
        height=720,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )

if __name__ == "__main__":
    from feature_extraction import extract_features_from_images
    from feature_matching import match_image_pairs,match_sift_features,filter_matches_by_homography
    from initial_recon import estimate_pose, print_pose_info
    
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

    # 9. 所有相机位姿恢复完毕后，进行可视化
    visualize_scene_and_points(camera_poses, scene_points_3d, save_path="reconstruction_direct.png")
    visualize_with_open3d(camera_poses, scene_points_3d)

    # 10. 打印最终重建结果
    print("\n" + "="*60)
    print("3D RECONSTRUCTION SUMMARY")
    print("="*60)
    print(f"Total cameras: {len(camera_poses)}")
    for i, pose in enumerate(camera_poses):
        R_i = pose['R']
        t_i = pose['t'].flatten() if pose['t'].ndim > 1 else pose['t']
        cam_world = -R_i.T @ t_i
        euler_i = np.degrees(cv2.Rodrigues(R_i)[0].flatten())
        print(f"\n  Camera {i} ({pose['image']}):")
        print(f"    - Position (world): {cam_world}")
        print(f"    - Rotation (Euler deg): {euler_i}")
    print(f"\nTotal 3D points: {scene_points_3d.shape[0]}")
    print("="*60)