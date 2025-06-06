#!/usr/bin/env python3
"""
三维重建主程序
使用特征提取和匹配模块进行三维重建的完整流程
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict
import time

# 导入自定义模块
try:
    from feature_extraction import (
        extract_features_from_images, 
        display_keypoints_for_images,
        save_features_to_file
    )
    from feature_matching import (
        match_image_pairs,
        display_matches_for_pairs,
        filter_matches_by_homography
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure feature_extraction.py and feature_matching.py are in the same directory")
    sys.exit(1)

class ThreeDReconstruction:
    """三维重建主类"""
    
    def __init__(self, image_directory: str = None, image_paths: List[str] = None):
        """
        初始化三维重建系统
        
        Args:
            image_directory: 包含图片的目录路径
            image_paths: 图片路径列表（二选一）
        """
        self.image_paths = []
        self.feature_data = []
        self.matches_data = []
        self.method = 'sift'  # 默认使用SIFT
        
        if image_directory:
            self.load_images_from_directory(image_directory)
        elif image_paths:
            self.image_paths = image_paths
        else:
            print("Warning: No images provided. Use load_images_from_directory() or set_image_paths()")
    
    def load_images_from_directory(self, directory: str, extensions: List[str] = None):
        """
        从目录加载图片
        
        Args:
            directory: 图片目录路径
            extensions: 支持的图片扩展名
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist!")
            return
        
        self.image_paths = []
        for filename in sorted(os.listdir(directory)):
            if any(filename.lower().endswith(ext) for ext in extensions):
                self.image_paths.append(os.path.join(directory, filename))
        
        print(f"Loaded {len(self.image_paths)} images from {directory}")
        for i, path in enumerate(self.image_paths):
            print(f"  {i+1}: {os.path.basename(path)}")
    
    def set_image_paths(self, image_paths: List[str]):
        """设置图片路径列表"""
        self.image_paths = image_paths
        print(f"Set {len(self.image_paths)} image paths")
    
    def set_feature_method(self, method: str):
        """
        设置特征提取方法
        
        Args:
            method: 'sift' 或 'orb'
        """
        if method.lower() in ['sift', 'orb']:
            self.method = method.lower()
            print(f"Feature extraction method set to: {self.method.upper()}")
        else:
            print(f"Unknown method: {method}. Using SIFT as default.")
            self.method = 'sift'
    
    def extract_features(self, display_keypoints: bool = True, max_display_images: int = 5):
        """
        步骤1：提取所有图片的特征
        
        Args:
            display_keypoints: 是否显示关键点
            max_display_images: 最多显示几张图片的关键点
        """
        print("\n" + "="*60)
        print("STEP 1: FEATURE EXTRACTION")
        print("="*60)
        
        if not self.image_paths:
            print("Error: No images loaded!")
            return False
        
        start_time = time.time()
        
        # 提取特征
        self.feature_data = extract_features_from_images(self.image_paths, method=self.method)
        
        # 检查提取结果
        valid_features = sum(1 for _, desc in self.feature_data if desc is not None)
        print(f"\nFeature extraction completed in {time.time() - start_time:.2f} seconds")
        print(f"Valid features extracted from {valid_features}/{len(self.image_paths)} images")
        
        if valid_features < 2:
            print("Error: Need at least 2 images with valid features for matching!")
            return False
        
        # 显示关键点（如果需要）
        if display_keypoints:
            display_keypoints_for_images(
                self.image_paths, 
                self.feature_data, 
                max_images=max_display_images, 
                save_visualizations=True
            )
        
        # 保存特征到文件
        save_features_to_file(self.feature_data, self.image_paths, f"{self.method}_features")
        
        return True
    
    def match_features(self, display_matches: bool = True, max_display_pairs: int = 3):
        """
        步骤2：匹配图片间的特征
        
        Args:
            display_matches: 是否显示匹配结果
            max_display_pairs: 最多显示几对图片的匹配
        """
        print("\n" + "="*60)
        print("STEP 2: FEATURE MATCHING")
        print("="*60)
        
        if not self.feature_data:
            print("Error: No features extracted! Run extract_features() first.")
            return False
        
        start_time = time.time()
        
        # 进行特征匹配
        self.matches_data = match_image_pairs(self.feature_data, method=self.method)
        
        # 统计匹配结果
        total_pairs = 0
        successful_pairs = 0
        total_matches = 0
        
        for i in range(len(self.image_paths)):
            for j in range(i+1, len(self.image_paths)):
                total_pairs += 1
                if i < len(self.matches_data) and j < len(self.matches_data[i]):
                    matches = self.matches_data[i][j]
                    if matches:
                        successful_pairs += 1
                        total_matches += len(matches)
        
        print(f"\nFeature matching completed in {time.time() - start_time:.2f} seconds")
        print(f"Successful matching pairs: {successful_pairs}/{total_pairs}")
        print(f"Total matches found: {total_matches}")
        
        # 显示匹配结果（如果需要）
        if display_matches and successful_pairs > 0:
            display_matches_for_pairs(
                self.image_paths, 
                self.feature_data, 
                self.matches_data, 
                max_pairs=max_display_pairs, 
                method=self.method
            )
        
        return True
    
    def filter_matches_with_homography(self, min_matches: int = 10):
        """
        步骤3：使用单应性矩阵过滤匹配点
        
        Args:
            min_matches: 最少匹配点数量
        """
        print("\n" + "="*60)
        print("STEP 3: HOMOGRAPHY FILTERING")
        print("="*60)
        
        if not self.matches_data:
            print("Error: No matches found! Run match_features() first.")
            return False
        
        filtered_matches = []
        homographies = []
        
        for i in range(len(self.image_paths)):
            filtered_row = []
            homography_row = []
            
            for j in range(len(self.image_paths)):
                if i == j:
                    filtered_row.append([])
                    homography_row.append(None)
                    continue
                
                if i < len(self.matches_data) and j < len(self.matches_data[i]):
                    matches = self.matches_data[i][j]
                    
                    if len(matches) >= min_matches:
                        kp1, _ = self.feature_data[i]
                        kp2, _ = self.feature_data[j]
                        
                        filtered, homography = filter_matches_by_homography(kp1, kp2, matches)
                        filtered_row.append(filtered)
                        homography_row.append(homography)
                        
                        if filtered:
                            print(f"Pair ({i+1}, {j+1}): {len(filtered)}/{len(matches)} matches after filtering")
                    else:
                        filtered_row.append([])
                        homography_row.append(None)
                else:
                    filtered_row.append([])
                    homography_row.append(None)
            
            filtered_matches.append(filtered_row)
            homographies.append(homography_row)
        
        # 更新匹配数据
        self.matches_data = filtered_matches
        self.homographies = homographies
        
        return True
    
    def get_match_statistics(self):
        """获取匹配统计信息"""
        if not self.matches_data:
            return None
        
        stats = {
            'total_images': len(self.image_paths),
            'total_pairs': 0,
            'successful_pairs': 0,
            'total_matches': 0,
            'avg_matches_per_pair': 0,
            'pair_details': []
        }
        
        for i in range(len(self.image_paths)):
            for j in range(i+1, len(self.image_paths)):
                stats['total_pairs'] += 1
                
                if i < len(self.matches_data) and j < len(self.matches_data[i]):
                    matches = self.matches_data[i][j]
                    if matches:
                        stats['successful_pairs'] += 1
                        stats['total_matches'] += len(matches)
                        stats['pair_details'].append({
                            'pair': (i+1, j+1),
                            'matches': len(matches),
                            'images': (os.path.basename(self.image_paths[i]), 
                                     os.path.basename(self.image_paths[j]))
                        })
        
        if stats['successful_pairs'] > 0:
            stats['avg_matches_per_pair'] = stats['total_matches'] / stats['successful_pairs']
        
        return stats
    
    def print_summary(self):
        """打印处理结果摘要"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        print(f"Total images processed: {len(self.image_paths)}")
        print(f"Feature extraction method: {self.method.upper()}")
        
        # 特征提取统计
        if self.feature_data:
            valid_features = sum(1 for _, desc in self.feature_data if desc is not None)
            print(f"Images with valid features: {valid_features}/{len(self.image_paths)}")
            
            for i, (kp, desc) in enumerate(self.feature_data):
                if desc is not None:
                    print(f"  Image {i+1}: {len(kp)} keypoints")
        
        # 匹配统计
        stats = self.get_match_statistics()
        if stats:
            print(f"\nMatching results:")
            print(f"  Successful pairs: {stats['successful_pairs']}/{stats['total_pairs']}")
            print(f"  Total matches: {stats['total_matches']}")
            print(f"  Average matches per pair: {stats['avg_matches_per_pair']:.1f}")
            
            if stats['pair_details']:
                print(f"\nTop matching pairs:")
                sorted_pairs = sorted(stats['pair_details'], key=lambda x: x['matches'], reverse=True)
                for pair_info in sorted_pairs[:5]:  # 显示前5个最佳匹配对
                    print(f"  Pair {pair_info['pair']}: {pair_info['matches']} matches "
                          f"({pair_info['images'][0]} ↔ {pair_info['images'][1]})")
    
    def run_full_pipeline(self, display_keypoints: bool = True, display_matches: bool = True,
                         max_display_images: int = 5, max_display_pairs: int = 3,
                         use_homography_filtering: bool = True):
        """
        运行完整的特征提取和匹配流程
        
        Args:
            display_keypoints: 是否显示关键点
            display_matches: 是否显示匹配结果
            max_display_images: 最多显示几张图片的关键点
            max_display_pairs: 最多显示几对图片的匹配
            use_homography_filtering: 是否使用单应性过滤
        """
        print("Starting 3D Reconstruction Pipeline...")
        print(f"Processing {len(self.image_paths)} images using {self.method.upper()} features")
        
        total_start_time = time.time()
        
        # 步骤1：特征提取
        if not self.extract_features(display_keypoints, max_display_images):
            print("Feature extraction failed!")
            return False
        
        # 步骤2：特征匹配
        if not self.match_features(display_matches, max_display_pairs):
            print("Feature matching failed!")
            return False
        
        # 步骤3：单应性过滤（可选）
        if use_homography_filtering:
            self.filter_matches_with_homography()
        
        total_time = time.time() - total_start_time
        print(f"\nPipeline completed in {total_time:.2f} seconds")
        
        # 打印摘要
        self.print_summary()
        
        return True

def main():
    """主函数 - 使用示例"""
    print("=== 3D Reconstruction Pipeline ===")
    
    # 方法1：从目录加载图片
    # reconstruction = ThreeDReconstruction(image_directory="./images")
    
    # 方法2：指定图片路径列表
    image_paths = [
        "images/DJI_20200223_163016_842.jpg",
        "images/DJI_20200223_163017_967.jpg",
        "images/DJI_20200223_163018_942.jpg"
        
    ]
    
    # 检查图片是否存在，如果不存在则创建测试图片
    import cv2
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"Creating test image: {path}")
            # 创建带有不同特征的测试图片
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 添加一些共同特征
            cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
            cv2.circle(img, (300, 200), 40, (0, 0, 255), -1)
            cv2.line(img, (50, 300), (550, 300), (0, 255, 0), 3)
            
            # 添加一些独特特征
            cv2.rectangle(img, (50+i*80, 50), (100+i*80, 100), (255, 0, 255), -1)
            cv2.circle(img, (400+i*30, 350), 25, (255, 255, 0), -1)
            
            # 添加一些噪声点
            for _ in range(20):
                x, y = np.random.randint(0, 640), np.random.randint(0, 480)
                cv2.circle(img, (x, y), 5, (np.random.randint(0, 255), 
                                           np.random.randint(0, 255), 
                                           np.random.randint(0, 255)), -1)
            
            cv2.imwrite(path, img)
    
    # 创建重建对象
    reconstruction = ThreeDReconstruction(image_paths=image_paths)
    
    # 设置特征提取方法
    reconstruction.set_feature_method('sift')  # 或 'orb'
    
    # 运行完整流程
    success = reconstruction.run_full_pipeline(
        display_keypoints=True,      # 显示关键点
        display_matches=True,        # 显示匹配结果
        max_display_images=3,        # 最多显示3张图片的关键点
        max_display_pairs=3,         # 最多显示3对图片的匹配
        use_homography_filtering=True # 使用单应性过滤
    )
    
    if success:
        print("\n=== Pipeline completed successfully! ===")
        print("Generated files:")
        print("- keypoints_*.jpg: Keypoint visualizations")
        print("- matches_*.jpg: Match visualizations")
        print("- *_features/: Saved feature data")
    else:
        print("\n=== Pipeline failed! ===")
    
    # 也可以分步骤运行
    # reconstruction.extract_features(display_keypoints=True)
    # reconstruction.match_features(display_matches=True)
    # reconstruction.filter_matches_with_homography()
    # reconstruction.print_summary()

if __name__ == "__main__":
    main()