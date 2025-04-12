# sfm_pipeline.py
from helper_functions import (load_coil_images, preprocess_image, draw_3d_point_cloud,
                              triangulate_pair, filter_point_cloud, interpolate_points)#,
                              #get_camera_pose)
import cv2
import numpy as np
import glob

def run_sfm_pipeline(coil_pattern, use_orb=True, interpolate_flag=True, filter_flag=True):
    """
    Run the complete SfM pipeline with improved feature detection and post-processing on COIL-100 images.
    Uses known turntable geometry for camera pose estimation.
    """
    # 1. Load the images matching the pattern
    images = load_coil_images(coil_pattern)
    print(f"Loaded {len(images)} images matching pattern {coil_pattern}")
    if len(images) < 2:
        raise ValueError("Need at least 2 images for SfM.")
        
    total_imgs = len(images)
    
    # Display the first image for verification
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    plt.title("First Image (Verification)")
    plt.axis('off')
    plt.show()
    
    # 2. Setup camera intrinsics: Estimate focal length as 1.5 * image width
    h, w = images[0].shape[:2]
    focal_length = 1.5 * w
    cx, cy = w / 2, h / 2
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]])
    print("Camera intrinsic matrix K:")
    print(K)
    
    # 3. Compute keypoints and descriptors for every image using ORB or SIFT
    if use_orb:
        detector = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        print("Computing ORB features for each image...")
    else:
        detector = cv2.SIFT_create()
        print("Computing SIFT features for each image...")
    
    keypoints_list = []
    descriptors_list = []
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = detector.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
        print(f"Image {idx}: {len(kp)} keypoints")
    
    # 4. Setup matcher for feature matching
    if use_orb:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 5. Use known turntable poses from COIL-100
    global_R = []
    global_t = []
    for i in range(total_imgs):
        R, t = get_camera_pose(i, total_images=total_imgs, radius=1.0)
        global_R.append(R)
        global_t.append(t)
    
    # 6. Incrementally triangulate points from consecutive pairs
    sparse_points = []
    used_frame_indices = [0]  # Start with the first image index
    current_idx = 0
    
    for i in range(1, total_imgs):
        matched = False
        for prev_idx in range(max(0, current_idx - 5), current_idx + 1):
            if (len(descriptors_list[prev_idx]) == 0 or len(descriptors_list[i]) == 0 or
                prev_idx not in used_frame_indices):
                continue
            
            pts1 = []
            pts2 = []
            # Perform matching with ratio test
            try:
                matches = matcher.knnMatch(descriptors_list[prev_idx], descriptors_list[i], k=2)
                good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
            except:
                good_matches = sorted(matcher.match(descriptors_list[prev_idx], descriptors_list[i]), key=lambda x: x.distance)[:50]
            
            for m in good_matches:
                pts1.append(keypoints_list[prev_idx][m.queryIdx].pt)
                pts2.append(keypoints_list[i][m.trainIdx].pt)
                
            pts1 = np.array(pts1, dtype=np.float32)
            pts2 = np.array(pts2, dtype=np.float32)
            
            if len(pts1) < 8 or len(pts2) < 8:
                print(f"Not enough matches between frame {prev_idx} and {i}: {len(pts1)} points")
                continue
            
            # Estimate the Essential matrix and recover pose
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 3.0, 0.99)
            if F is None or F.shape != (3, 3):
                print(f"Fundamental matrix estimation failed between frame {prev_idx} and {i}")
                continue
            mask = mask.ravel().astype(bool)
            pts1 = pts1[mask]
            pts2 = pts2[mask]
            if len(pts1) < 8:
                print(f"Not enough inliers between frame {prev_idx} and {i}: {len(pts1)} points")
                continue
            E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
            if E is None or E.shape != (3, 3):
                print(f"Essential matrix estimation failed between frame {prev_idx} and {i}")
                continue
            
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
            R_prev, t_prev = global_R[used_frame_indices.index(prev_idx)], global_t[used_frame_indices.index(prev_idx)]
            R_global = R_prev @ R
            t_global = R_prev @ t + t_prev
            
            proj1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
            proj2 = K @ np.hstack((R, t))
            pts3d_local = triangulate_pair(K, proj1, proj2, pts1, pts2)
            valid_points = sum(1 for pt in pts3d_local if pt[2] > 0 and (R @ pt + t.ravel())[2] > 0)
            if valid_points < 0.7 * len(pts3d_local):
                print(f"Too many points behind cameras between frame {prev_idx} and {i}: {valid_points}/{len(pts3d_local)}")
                continue
            
            pts3d_global = (R_prev @ pts3d_local.T + t_prev).T
            global_R.append(R_global)
            global_t.append(t_global)
            used_frame_indices.append(i)
            sparse_points.append(pts3d_global)
            print(f"Processed frame pair {prev_idx}-{i}: triangulated {pts3d_local.shape[0]} points")
            matched = True
            current_idx = i
            break
        
        if not matched:
            print(f"Could not match frame {i} with any previous frame")
    
    if sparse_points:
        sparse_points = np.vstack(sparse_points)
    else:
        sparse_points = np.empty((0, 3))
    
    print(f"Sparse SfM point cloud: {sparse_points.shape[0]} points")
    if filter_flag and sparse_points.shape[0] > 20:
        sparse_points = filter_point_cloud(sparse_points)
    if interpolate_flag and sparse_points.shape[0] > 4:
        sparse_points = interpolate_points(sparse_points)
    
    draw_3d_point_cloud(sparse_points, "Final SfM Reconstruction (Sparse + Interpolated)", point_size=2, color='r')
    print("SfM reconstruction completed.")
    print("Used frames indices:", used_frame_indices)
    return sparse_points, global_R, global_t, used_frame_indices
