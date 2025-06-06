# Scene Initialization using Epipolar Geometry
# Estimates relative pose between two cameras for 3D reconstruction initialization
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
            