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

def extract_sift_features(image_path: str, max_features: int = 10000) -> Tuple[List, Optional[np.ndarray]]:
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
        visualize_keypoints(image_paths[i], keypoints, max_keypoints=2500, save_path=save_path)

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
        "images/DJI_20200223_163017_967.jpg",
        "images/DJI_20200223_163018_942.jpg" 
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
    
    sift_features = extract_features_from_images(image_paths, method='sift')
        
    
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