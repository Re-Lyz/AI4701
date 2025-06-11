import numpy as np
import cv2
import os
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

'''
Feature Matching Module
1.match_sift_features: Matches SIFT features using FLANN and Lowe's ratio test. 
2.match_features_brute_force: Fallback brute force matcher.
3.match_image_pairs: Matches features between all pairs of images.
4.visualize_matches: Visualizes feature matches between two images.
5.display_matches_for_pairs: Displays matches for the first few image pairs.
6.filter_matches_by_homography: Filters matches using RANSAC homography estimation.

'''

def match_sift_features(desc1: np.ndarray, desc2: np.ndarray, ratio_threshold: float = 0.7) -> List:
    """
    Match SIFT features between two images using Lowe's ratio test.
    
    Args:
        desc1, desc2: SIFT descriptors from two images
        ratio_threshold: Ratio threshold for Lowe's ratio test
        
    Returns:
        List of good matches
    """
    if desc1 is None or desc2 is None:
        print("Error: One or both descriptor arrays are None!")
        return []
    
    print(f"Matching features: {desc1.shape[0]} vs {desc2.shape[0]} descriptors")
    
    # Create FLANN matcher for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        # Find k=2 nearest neighbors
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches out of {len(matches)} total matches")
        print(f"Match ratio: {len(good_matches)/len(matches)*100:.1f}%")
        
        # Print distance statistics
        if good_matches:
            distances = [m.distance for m in good_matches]
            print(f"Distance stats - Min: {min(distances):.2f}, Max: {max(distances):.2f}, Mean: {np.mean(distances):.2f}")
        
        return good_matches
        
    except Exception as e:
        print(f"Error in FLANN matching: {e}")
        # Fallback to BFMatcher
        return match_features_brute_force(desc1, desc2)

def match_features_brute_force(desc1: np.ndarray, desc2: np.ndarray) -> List:
    """
    Brute force feature matching (fallback method).
    
    Args:
        desc1, desc2: Descriptors from two images
        
    Returns:
        List of matches
    """
    print("Using brute force matcher as fallback...")
    
    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Match features
    matches = bf.match(desc1, desc2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"Brute force found {len(matches)} matches")
    
    return matches

def match_image_pairs(feature_data: List[Tuple], method: str = 'sift') -> List[List]:
    """
    Match features between all pairs of images.
    
    Args:
        feature_data: List of (keypoints, descriptors) tuples
        method: Matching method ('sift' or 'orb')
        
    Returns:
        List of match results for each image pair
    """
    print(f"\n=== Starting pairwise feature matching using {method.upper()} ===")
    
    n_images = len(feature_data)
    all_matches = []
    
    for i in range(n_images):
        matches_for_image_i = []
        for j in range(n_images):
            if i == j:
                matches_for_image_i.append([])  # No self-matching
                continue
                
            print(f"\n--- Matching image {i+1} with image {j+1} ---")
            
            _, desc1 = feature_data[i]
            _, desc2 = feature_data[j]
            
            if desc1 is None or desc2 is None:
                print(f"Skipping pair ({i+1}, {j+1}) - missing descriptors")
                matches_for_image_i.append([])
                continue
            
            if method.lower() == 'sift':
                matches = match_sift_features(desc1, desc2)
            else:
                print(f"Unknown method: {method}. Using SIFT as default.")
                matches = match_sift_features(desc1, desc2)
            
            matches_for_image_i.append(matches)
            
        all_matches.append(matches_for_image_i)
    
    # Print summary
    print(f"\n=== Matching Summary ===")
    total_pairs = 0
    successful_pairs = 0
    
    for i in range(n_images):
        for j in range(i+1, n_images):  # Only count each pair once
            total_pairs += 1
            if len(all_matches[i][j]) > 0:
                successful_pairs += 1
                print(f"Pair ({i+1}, {j+1}): {len(all_matches[i][j])} matches")
    
    print(f"Successful matching pairs: {successful_pairs}/{total_pairs}")
    
    return all_matches

def visualize_matches(img1_path: str, img2_path: str, kp1: List, kp2: List, matches: List, 
                     max_matches: int = 50, save_path: Optional[str] = None, show_plot: bool = True, save: bool = False):
    """
    Visualize feature matches between two images.
    
    Args:
        img1_path, img2_path: Paths to the two images
        kp1, kp2: Keypoints from both images
        matches: List of matches
        max_matches: Maximum number of matches to display
        save_path: Path to save the visualization
        show_plot: Whether to display the plot
    """
    if not matches:
        print("No matches to visualize!")
        return
    
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Cannot read one or both images for visualization!")
        return
    
    # Limit number of matches for cleaner visualization
    matches_to_show = matches[:max_matches]
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches_to_show, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    print(f"Visualizing {len(matches_to_show)} matches out of {len(matches)} total matches")
    
    if save_path is not None and save:
        cv2.imwrite(save_path, img_matches)
        print(f"Match visualization saved to: {save_path}")
    
    # Display using matplotlib if available and requested
    if show_plot:
        try:
            plt.figure(figsize=(15, 8))
            plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            plt.title(f'Feature Matches: {os.path.basename(img1_path)} ↔ {os.path.basename(img2_path)} ({len(matches_to_show)} matches)')
            plt.axis('off')
            if save:
                plt.savefig(save_path.replace('.jpg', '_plt.png'), dpi=150, bbox_inches='tight')
            plt.show()
        except ImportError:
            print("Matplotlib not available for display. Image saved to file if save_path provided.")

def display_matches_for_pairs(image_paths: List[str], feature_data: List[Tuple], 
                            all_matches: List[List], max_pairs: int = 3, method: str = 'sift'):
    """
    Display match visualizations for the first few image pairs.
    
    Args:
        image_paths: List of image paths
        feature_data: List of (keypoints, descriptors) tuples
        all_matches: All pairwise matches
        max_pairs: Maximum number of pairs to visualize
        method: Feature extraction method used
    """
    print(f"\n=== Displaying matches for first {max_pairs} image pairs ===")
    
    pair_count = 0
    for i in range(len(image_paths)):
        for j in range(i+1, len(image_paths)):
            if pair_count >= max_pairs:
                return
                
            matches = all_matches[i][j] if i < len(all_matches) and j < len(all_matches[i]) else []
            
            if not matches:
                print(f"\nPair ({i+1}, {j+1}): No matches to display")
                continue
                
            print(f"\n--- Displaying matches for pair ({i+1}, {j+1}) ---")
            print(f"Images: {os.path.basename(image_paths[i])} ↔ {os.path.basename(image_paths[j])}")
            print(f"Number of matches: {len(matches)}")
            
            kp1, _ = feature_data[i]
            kp2, _ = feature_data[j]
            
            # Generate save path
            base1 = os.path.splitext(os.path.basename(image_paths[i]))[0]
            base2 = os.path.splitext(os.path.basename(image_paths[j]))[0]
            save_path = f"matches_{method}_{base1}_to_{base2}.jpg"
            
            # Visualize matches
            visualize_matches(image_paths[i], image_paths[j], kp1, kp2, matches,
                            max_matches=50, save_path=save_path, show_plot=True)
            
            pair_count += 1

def filter_matches_by_homography(kp1: List, kp2: List, matches: List, 
                                ransac_threshold: float = 5.0) -> Tuple[List, Optional[np.ndarray]]:
    """
    Filter matches using RANSAC homography estimation.
    
    Args:
        kp1, kp2: Keypoints from both images
        matches: Initial matches
        ransac_threshold: RANSAC threshold for homography estimation
        
    Returns:
        Tuple of (filtered_matches, homography_matrix)
    """
    if len(matches) < 4:
        print("Not enough matches for homography estimation (need at least 4)")
        return matches, None
    
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    try:
        homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                            cv2.RANSAC, ransac_threshold)
        
        # Filter matches based on inliers
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        
        print(f"Homography filtering: {len(inlier_matches)} inliers out of {len(matches)} matches")
        print(f"Inlier ratio: {len(inlier_matches)/len(matches)*100:.1f}%")
        
        return inlier_matches, homography
        
    except Exception as e:
        print(f"Error in homography estimation: {e}")
        return matches, None

if __name__ == "__main__":
    # Import the feature extraction module
    try:
        from feature_extraction import extract_features_from_images
    except ImportError:
        print("Error: Cannot import feature_extraction module!")
        print("Make sure feature_extraction.py is in the same directory.")
        exit(1)
    
    print("=== Feature Matching Module Test ===")
    
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
    
    # Extract features using SIFT
    print("\n" + "="*60)
    print("EXTRACTING FEATURES FOR MATCHING TEST")
    print("="*60)
    
    sift_features = extract_features_from_images(image_paths, method='sift')
    
    # Test SIFT matching
    print("\n" + "="*60)
    print("TESTING SIFT FEATURE MATCHING")
    print("="*60)
    
    sift_matches = match_image_pairs(sift_features, method='sift')
    
    # Display matches for the first few pairs
    display_matches_for_pairs(image_paths, sift_features, sift_matches, max_pairs=3, method='sift')
    
    # Test specific pair matching with visualization
    if len(sift_features) >= 2:
        kp1, desc1 = sift_features[0]
        kp2, desc2 = sift_features[1]
        
        if desc1 is not None and desc2 is not None:
            print(f"\n--- Detailed analysis of pair (1, 2) ---")
            matches_1_2 = match_sift_features(desc1, desc2)
            
            # Apply homography filtering
            if len(matches_1_2) >= 4:
                filtered_matches, homography = filter_matches_by_homography(kp1, kp2, matches_1_2)
                
                # Visualize original and filtered matches
                print(f"\n=== Detailed Match Visualization for Pair (1, 2) ===")
                visualize_matches(image_paths[0], image_paths[1], kp1, kp2, 
                                matches_1_2, max_matches=30, save_path="matches_original.jpg", show_plot=False)
                
                if len(filtered_matches) > 0:
                    visualize_matches(image_paths[0], image_paths[1], kp1, kp2, 
                                    filtered_matches, max_matches=30, save_path="matches_filtered.jpg", show_plot=False)

    # Final summary
    print("\n" + "="*60)
    print("MATCHING TEST SUMMARY")
    print("="*60)
    
    for i in range(len(image_paths)):
        for j in range(i+1, len(image_paths)):
            sift_count = len(sift_matches[i][j]) if i < len(sift_matches) and j < len(sift_matches[i]) else 0
            print(f"Pair ({i+1}, {j+1}): SIFT={sift_count} matches")
    
    print("\nFeature matching test completed!")