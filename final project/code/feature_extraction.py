import numpy as np
import cv2
import os
from typing import List, Tuple, Optional
from pathlib import Path

"""
Extract SIFT features from an image.

主要包含以下几个函数功能：
1. `extract_sift_features`: Extracts SIFT features from a single image. 返回一个包含关键点和描述符的元组。
2. `extract_features_from_images`: Extracts features from a list of images using specified method (SIFT or ORB). 返回一个包含每个图像的关键点和描述符的列表。
3. `visualize_keypoints`: Visualizes keypoints on an image and optionally saves the visualization. 返回 None，用于显示图片。
4. `display_keypoints_for_images`: Displays keypoints for the first few images in the list. 返回 None，用于显示图片。
5. `save_features_to_file`: Saves extracted features to files for later use. 返回 None，用于保存特征到文件。

"""

def extract_sift_features_colmap_style(
    image_path: str,
    max_features: int = 8192,
    grid_size: int = 4,
    contrast_threshold: float = 0.02,
    edge_threshold: float = 10.0,
    enable_rootsift: bool = True,
    enable_subpixel_refinement: bool = True
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    COLMAP风格的SIFT特征提取，完全模拟COLMAP的特征提取策略:
    
    1. 网格化截断（Spatial Binning）：4×4网格均匀分布
    2. 高特征上限：默认8192个特征点
    3. 灵敏DoG参数：contrast_threshold=0.02, edge_threshold=10
    4. RootSIFT归一化：L1归一化后开方，对光照变化更鲁棒
    5. 子像素优化：精确定位关键点位置
    6. 多尺度检测：确保尺度不变性
    
    Args:
        image_path: 图像文件路径
        max_features: 总特征点数上限
        grid_size: 网格大小（每个轴上的分割数）
        contrast_threshold: DoG对比度阈值，越小越敏感
        edge_threshold: 边缘阈值，用于抑制边缘响应
        enable_rootsift: 是否启用RootSIFT归一化
        enable_subpixel_refinement: 是否启用子像素优化
    
    Returns:
        keypoints: 关键点列表
        descriptors: 描述子数组 (N, 128) 或 None
    """
    
    # 输入验证
    if not os.path.exists(image_path):
        print(f"错误: 图像路径 {image_path} 不存在!")
        return [], None
    
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return [], None
    
    # 启用多线程加速
    cv2.setNumThreads(os.cpu_count())
    
    h, w = img.shape
    print(f"图像尺寸: {w}x{h}")
    
    # 创建SIFT检测器，使用COLMAP相似的参数
    sift = cv2.SIFT_create(
        nfeatures=0,  # 先不限制总数，让每个网格自由检测
        nOctaveLayers=3,  # 默认3层
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=1.6  # 高斯核标准差
    )
    
    # 计算网格参数
    total_cells = grid_size * grid_size
    target_per_cell = max_features // total_cells
    
    print(f"网格化参数: {grid_size}x{grid_size} = {total_cells}个单元")
    print(f"每个单元目标特征数: {target_per_cell}")
    
    # 计算每个网格单元的尺寸
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    all_keypoints = []
    cell_results = []
    
    # 为每个网格单元检测特征
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算当前单元的边界
            y_start = i * cell_h
            y_end = (i + 1) * cell_h if i < grid_size - 1 else h
            x_start = j * cell_w  
            x_end = (j + 1) * cell_w if j < grid_size - 1 else w
            
            # 创建该单元的掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y_start:y_end, x_start:x_end] = 255
            
            # 在当前单元中检测关键点
            cell_keypoints = sift.detect(img, mask)
            
            if len(cell_keypoints) == 0:
                continue
                
            # 将tuple转换为list，然后按响应强度排序
            cell_keypoints = list(cell_keypoints)
            cell_keypoints.sort(key=lambda kp: kp.response, reverse=True)

            
            # 限制每个单元的特征点数量
            selected_keypoints = cell_keypoints[:target_per_cell]
            
            # 子像素优化（如果启用）
            if enable_subpixel_refinement and len(selected_keypoints) > 0:
                selected_keypoints = _refine_keypoints_subpixel(img, selected_keypoints)
            
            all_keypoints.extend(selected_keypoints)
            cell_results.append((i, j, len(selected_keypoints)))
    
    # 打印网格分布统计
    print("各网格单元特征点分布:")
    for i, j, count in cell_results:
        print(f"  单元[{i},{j}]: {count}个特征点")
    
    if len(all_keypoints) == 0:
        print("警告: 未检测到任何特征点")
        return [], None
    
    print(f"总共检测到 {len(all_keypoints)} 个特征点")
    
    # 计算描述子
    final_keypoints, descriptors = sift.compute(img, all_keypoints)
    
    if descriptors is None:
        print("警告: 描述子计算失败")
        return final_keypoints, None
    
    # 应用RootSIFT归一化（如果启用）
    if enable_rootsift:
        descriptors = _apply_rootsift_normalization(descriptors)
        print("已应用RootSIFT归一化")
    
    # 最终检查：确保特征数量不超过限制
    if len(final_keypoints) > max_features:
        # 按响应强度重新排序并截断
        keypoint_response_pairs = [(kp, desc) for kp, desc in zip(final_keypoints, descriptors)]
        keypoint_response_pairs.sort(key=lambda x: x[0].response, reverse=True)
        
        final_keypoints = [pair[0] for pair in keypoint_response_pairs[:max_features]]
        descriptors = np.array([pair[1] for pair in keypoint_response_pairs[:max_features]])
        
        print(f"最终截断到 {len(final_keypoints)} 个特征点")
    
    return final_keypoints, descriptors

def _refine_keypoints_subpixel(img: np.ndarray, keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
    """
    子像素级关键点优化，提高定位精度
    """
    refined_keypoints = []
    
    for kp in keypoints:
        x, y = kp.pt
        
        # 确保在图像边界内
        if x < 1 or x >= img.shape[1] - 1 or y < 1 or y >= img.shape[0] - 1:
            refined_keypoints.append(kp)
            continue
        
        # 计算梯度
        ix = int(x)
        iy = int(y)
        
        # 使用二次插值优化位置
        try:
            # 提取3x3窗口
            window = img[iy-1:iy+2, ix-1:ix+2].astype(np.float32)
            
            if window.shape != (3, 3):
                refined_keypoints.append(kp)
                continue
            
            # 计算梯度和Hessian矩阵
            dx = (window[1, 2] - window[1, 0]) / 2.0
            dy = (window[2, 1] - window[0, 1]) / 2.0
            
            dxx = window[1, 0] - 2*window[1, 1] + window[1, 2]
            dyy = window[0, 1] - 2*window[1, 1] + window[2, 1]
            dxy = (window[0, 0] - window[0, 2] - window[2, 0] + window[2, 2]) / 4.0
            
            # 构建Hessian矩阵
            H = np.array([[dxx, dxy], [dxy, dyy]])
            gradient = np.array([dx, dy])
            
            # 求解位置偏移
            if np.linalg.det(H) != 0:
                offset = -np.linalg.solve(H, gradient)
                
                # 限制偏移量
                if abs(offset[0]) < 0.5 and abs(offset[1]) < 0.5:
                    refined_kp = cv2.KeyPoint(
                        x + offset[0], 
                        y + offset[1], 
                        kp.size, 
                        kp.angle, 
                        kp.response, 
                        kp.octave, 
                        kp.class_id
                    )
                    refined_keypoints.append(refined_kp)
                else:
                    refined_keypoints.append(kp)
            else:
                refined_keypoints.append(kp)
                
        except Exception:
            refined_keypoints.append(kp)
    
    return refined_keypoints

def _apply_rootsift_normalization(descriptors: np.ndarray) -> np.ndarray:
    """
    应用RootSIFT归一化：L1归一化后开方
    这种归一化对光照变化和曝光变化更加鲁棒
    """
    descriptors = descriptors.astype(np.float32)
    
    # L1归一化
    l1_norms = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True)
    l1_norms = np.maximum(l1_norms, 1e-7)  # 避免除零
    descriptors = descriptors / l1_norms
    
    # 开方（Hellinger kernel）
    descriptors = np.sqrt(descriptors)
    
    return descriptors


def extract_sift_features_improved(
    image_path: str,
    max_features: int = 16384,
    grid_size: int = 4
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Improved SIFT feature extraction approximating COLMAP:

    - Grid-based uniform sampling: splits image into grid_size×grid_size cells
      and extracts up to max_features/grid_size^2 features per cell.
    - Increased feature cap: default max_features=8192.
    - Lower contrast threshold for DoG: contrastThreshold=0.02.
    - RootSIFT normalization: L1 normalize then sqrt.
    - Multithreading: uses all CPU cores.

    Args:
        image_path: Path to the image file.
        max_features: Total maximum number of features.
        grid_size: Number of spatial bins along each axis.

    Returns:
        keypoints: List of cv2.KeyPoint.
        descriptors: np.ndarray of shape (N,128) or None.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist!")
        return [], None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return [], None

    # Use all available threads
    cv2.setNumThreads(os.cpu_count())

    h, w = img.shape
    # Initialize SIFT with COLMAP-like parameters
    sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=0.02,
        edgeThreshold=10
    )

    # Determine per-cell feature limit
    total_cells = grid_size * grid_size
    per_cell = max_features // total_cells

    keypoints = []
    cell_h = int(np.ceil(h / grid_size))
    cell_w = int(np.ceil(w / grid_size))

    # Detect per-cell
    for i in range(grid_size):
        for j in range(grid_size):
            y0 = i * cell_h
            y1 = min((i + 1) * cell_h, h)
            x0 = j * cell_w
            x1 = min((j + 1) * cell_w, w)
            mask = np.zeros_like(img, dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255

            # Detect keypoints in this cell
            kps = sift.detect(img, mask)
            if not kps:
                continue
            # Keep strongest per cell
            kps = sorted(kps, key=lambda kp: kp.response, reverse=True)
            keypoints.extend(kps[:per_cell])

    if not keypoints:
        return [], None

    # Compute descriptors
    keypoints, descriptors = sift.compute(img, keypoints)

    # Apply RootSIFT
    if descriptors is not None:
        descriptors = descriptors.astype(np.float32)
        # L1 normalize
        norms = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True) + 1e-7
        descriptors = descriptors / norms
        # Hellinger kernel
        descriptors = np.sqrt(descriptors)

    return keypoints, descriptors

def extract_sift_features(image_path: str, max_features: int = 5000) -> Tuple[List, Optional[np.ndarray]]:
    """
    Extract SIFT features from an image.
    
    Args:
        image_path (str): Path to the image file
        max_features (int): Maximum number of features to extract
        
    Returns:
        Tuple[List, Optional[np.ndarray]]: Keypoints and descriptors
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist!")
        return [], None
        
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return [], None
        
    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector with maximum features limit
    sift = cv2.SIFT_create(nfeatures=max_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    print(f"Found {len(keypoints)} keypoints")
    if descriptors is not None:
        print(f"Descriptor shape: {descriptors.shape}")
        print(f"First few descriptor values: {descriptors[0][:5] if len(descriptors) > 0 else 'None'}")
    else:
        print("No descriptors found!")
    
    return keypoints, descriptors

def extract_features_from_images(image_paths: List[str], method: str = 'sift') -> List[Tuple[List, Optional[np.ndarray]]]:
    """
    Extract features from a list of images.
    
    Args:
        image_paths (List[str]): List of image file paths
        method (str): Feature extraction method ('sift' or 'orb')
        
    Returns:
        List[Tuple]: List of (keypoints, descriptors) tuples for each image
    """
    print(f"\n=== Starting feature extraction using {method.upper()} ===")
    feature_data = []
    
    for i, img_path in enumerate(image_paths):
        print(f"\n--- Processing image {i+1}/{len(image_paths)} ---")
        
        if method.lower() == 'sift':
            keypoints, descriptors = extract_sift_features(img_path)
            
        elif method.lower() == 'sift_improved':
            keypoints, descriptors = extract_sift_features_improved(img_path)
            
        elif method.lower() == 'sift_colmap':
            keypoints, descriptors = extract_sift_features_colmap_style(img_path)

        else:
            print(f"Unknown method: {method}. Using SIFT as default.")
            keypoints, descriptors = extract_sift_features(img_path)
            
        feature_data.append((keypoints, descriptors))
        

        if descriptors is None:
            print("Failed to extract features from this image")
    
    print(f"\n=== Feature extraction completed ===")
    valid_images = sum(1 for _, desc in feature_data if desc is not None)
    print(f"Valid images with features: {valid_images}/{len(image_paths)}")
    
    return feature_data

def load_or_extract_features(
    image_paths: List[str],
    method: str = 'sift',
    load: bool = True,
    save: bool = False,
    output_dir: Optional[str] = None
) -> List[Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]]:
    """
    load=True 时先检查 output_dir/features.yml，
      - 如果存在则一次性加载所有图像的特征；
      - 否则走 extract（并可选保存）。
    load=False 时直接走 extract。
    """
    if save and output_dir is None:
        raise ValueError("save=True 时必须指定 output_dir")
    if save:
        os.makedirs(output_dir, exist_ok=True)

    # 1) 如果允许加载并且文件存在，就一次性读回所有特征
    if load and output_dir:
        fs_path = Path(output_dir) / "features.yml"
        if fs_path.exists():
            fs = cv2.FileStorage(str(fs_path), cv2.FILE_STORAGE_READ)
            feats = []
            for idx in range(len(image_paths)):
                desc = fs.getNode(f"descriptors_{idx}").mat()
                kp_arr = fs.getNode(f"keypoints_{idx}").mat()
                kps = [
                    cv2.KeyPoint(
                        x=float(x), y=float(y),
                        size=float(size), angle=float(angle),
                        response=float(response),
                        octave=int(octave),
                        class_id=int(class_id)
                    )
                    for x, y, size, angle, response, octave, class_id in kp_arr
                ]
                feats.append((kps, desc))
            fs.release()
            print("检测到 features.yml，已一次性加载所有特征。")
            return feats

    # 2) 否则重新提取（并可选保存）
    print("开始提取特征…")
    return extract_features_from_images_save(
        image_paths, method=method, save=save, output_dir=output_dir
    )

def extract_features_from_images_save(
    image_paths: List[str],
    method: str = 'sift',
    save: bool = False,
    output_dir: Optional[str] = None
) -> List[Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]]:
    """
    对 image_paths 中所有图像做特征提取，返回 [(kps, desc), ...]。
    如果 save=True，会把所有结果写入 output_dir/features.yml。
    """
    feature_data = []
    for img_path in image_paths:
        if method.lower() == 'sift':
            kps, desc = extract_sift_features(img_path)
        elif method.lower() == 'sift_improved':
            kps, desc = extract_sift_features_improved(img_path)
        elif method.lower() == 'sift_colmap':
            kps, desc = extract_sift_features_colmap_style(img_path)
        else:
            print(f"Unknown method: {method}, fallback to SIFT.")
            kps, desc = extract_sift_features(img_path)
        feature_data.append((kps, desc))

    if save and output_dir:
        fs_path = Path(output_dir) / "features.yml"
        fs = cv2.FileStorage(str(fs_path), cv2.FILE_STORAGE_WRITE)
        for idx, (kps, desc) in enumerate(feature_data):
            # 写 descriptors（空矩阵也写）
            fs.write(f"descriptors_{idx}", desc if desc is not None else np.array([]))
            # 写 keypoints 为 (N,7) 数组
            kp_arr = np.array([
                (kp.pt[0], kp.pt[1],
                 kp.size, kp.angle,
                 kp.response, kp.octave, kp.class_id)
                for kp in kps
            ], dtype=np.float32)
            fs.write(f"keypoints_{idx}", kp_arr)
        fs.release()
        print(f"已保存所有图像特征到 {fs_path}")

    valid = sum(1 for _, d in feature_data if d is not None)
    print(f"=== 特征提取完成: {valid}/{len(image_paths)} 张图像有效 ===")
    return feature_data

def visualize_keypoints(image_path: str, keypoints: List, max_keypoints: int = 100, save_path: Optional[str] = None):
    """
    Visualize keypoints on an image.
    
    Args:
        image_path (str): Path to the image file
        keypoints (List): List of keypoints
        max_keypoints (int): Maximum number of keypoints to display
        save_path (Optional[str]): Path to save the visualization
    """
    if not os.path.exists(image_path):
        print(f"Error: Cannot find image {image_path} for visualization")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path} for visualization")
        return
    
    # Limit keypoints for cleaner visualization
    kp_to_show = keypoints[:max_keypoints] if len(keypoints) > max_keypoints else keypoints
    
    # Draw keypoints
    img_with_keypoints = cv2.drawKeypoints(img, kp_to_show, None, 
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    print(f"Displaying {len(kp_to_show)} keypoints out of {len(keypoints)} total")
    
    # Save visualization if path provided
    if save_path:
        cv2.imwrite(save_path, img_with_keypoints)
        print(f"Keypoint visualization saved to: {save_path}")
    
    # Display using matplotlib if available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title(f'Keypoints Visualization - {os.path.basename(image_path)} ({len(kp_to_show)} points)')
        plt.axis('off')
        plt.show()
    except ImportError:
        print("Matplotlib not available for display. Image saved to file if save_path provided.")

def display_keypoints_for_images(image_paths: List[str], feature_data: List[Tuple], 
                               max_images: int = 5, save_visualizations: bool = False):
    """
    Display keypoints for the first few images.
    
    Args:
        image_paths (List[str]): List of image paths
        feature_data (List[Tuple]): List of (keypoints, descriptors) tuples
        max_images (int): Maximum number of images to visualize
        save_visualizations (bool): Whether to save visualization images
    """
    print(f"\n=== Displaying keypoints for first {min(max_images, len(image_paths))} images ===")
    
    for i in range(min(max_images, len(image_paths))):
        keypoints, descriptors = feature_data[i]
        
        if descriptors is None:
            print(f"\nImage {i+1}: No keypoints to display (feature extraction failed)")
            continue
            
        print(f"\n--- Image {i+1}: {os.path.basename(image_paths[i])} ---")
        print(f"Total keypoints: {len(keypoints)}")
        
        # Generate save path if requested
        save_path = None
        if save_visualizations:
            base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
            save_path = f"keypoints_{base_name}.jpg"
        
        # Visualize keypoints
        visualize_keypoints(image_paths[i], keypoints, max_keypoints=5000, save_path=save_path)

def save_features_to_file(feature_data: List[Tuple[List, Optional[np.ndarray]]],
                          image_paths: List[str],
                          output_dir: str,
                        ):
    """
    Save extracted features to files for later use.
    
    Args:
        feature_data: List of (keypoints, descriptors) tuples
        image_paths: Original image paths
        output_dir: Directory to save feature files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n=== Saving features to {output_dir} ===")
    
    for i, ((keypoints, descriptors), img_path) in enumerate(zip(feature_data, image_paths)):
        if descriptors is not None:
            # Save descriptors
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            desc_file = os.path.join(output_dir, f"{base_name}_descriptors.npy")
            np.save(desc_file, descriptors)
            
            # Save keypoint coordinates
            kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            kp_file = os.path.join(output_dir, f"{base_name}_keypoints.npy")
            np.save(kp_file, kp_coords)
            
            print(f"Saved features for {base_name}: {len(keypoints)} keypoints")


if __name__ == "__main__":
    # Test the module
    print("=== Feature Extraction Module Test ===")
    
    # Example image paths (replace with your actual image paths)
    image_paths = [
        "images/DJI_20200223_163016_842.jpg",
    ]
    
    # Check if test images exist, if not create dummy ones for testing
    test_images_exist = all(os.path.exists(path) for path in image_paths)
    
    if not test_images_exist:
        print("Test images not found. Creating dummy test images...")
        # Create dummy test images
        for i, path in enumerate(image_paths):
            # Create a simple test image with some patterns
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add some geometric patterns for feature detection
            cv2.rectangle(img, (50+i*50, 50), (150+i*50, 150), (255, 255, 255), -1)
            cv2.circle(img, (300+i*30, 200), 30, (0, 0, 255), -1)
            cv2.line(img, (100, 300+i*20), (500, 350+i*20), (0, 255, 0), 3)
            cv2.imwrite(path, img)
        print("Dummy test images created successfully!")
    
    # Test SIFT feature extraction
    print("\n" + "="*50)
    print("Testing SIFT Feature Extraction")
    print("="*50)
    
    # sift_features = extract_features_from_images(image_paths, method='sift')
    sift_features = extract_features_from_images(image_paths, method='sift_colmap')
    
    # Display keypoints for the first 5 images
    display_keypoints_for_images(image_paths, sift_features, max_images=5, save_visualizations=False)
    
    # Save features
    save_features_to_file(sift_features, image_paths, "sift_features")
    
    # Print final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    for i, (path, (sift_kp, sift_desc))in enumerate(zip(image_paths, sift_features)):
        print(f"\nImage {i+1}: {path}")
        print(f"  SIFT: {len(sift_kp) if sift_desc is not None else 0} keypoints")
    
    print("\nFeature extraction test completed!")