# helper_functions.py
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage import measure  # For marching cubes

def load_coil_images(image_pattern):
    """
    Loads and sorts all images matching the given glob pattern.
    For example: "object21/obj21__*.png" for object 21.
    """
    image_paths = sorted(glob.glob(image_pattern))
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Unable to load {path}")
        else:
            # Preprocess images - enhance contrast
            img = preprocess_image(img)
            images.append(img)
    return images

def preprocess_image(img):
    """Enhanced preprocessing for COIL-100 images."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply background segmentation (assuming object is centered)
    h, w = gray.shape
    center = (w // 2, h // 2)
    radius = min(w, h) // 2 - 10  # Slightly smaller than half the image
    
    # Create a circular mask for the object
    mask = np.zeros_like(gray)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply mask to isolate object
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)
    
    # Apply sharpening
    blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.8, blur, -0.8, 0)
    
    # Convert back to BGR
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return result

def get_camera_pose(frame_idx, total_images=72, radius=1.0):
    """
    Calculate camera pose for turntable setup.
    
    Args:
        frame_idx: Index of the current frame (0 to total_images-1)
        total_images: Total number of images in a full 360° rotation
        radius: Distance from camera to object center
        
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    """
    # Calculate angle in radians
    theta = 2 * np.pi * frame_idx / total_images
    
    # Camera position in world coordinates
    cam_pos = np.array([
        radius * np.cos(theta),
        0,
        radius * np.sin(theta)
    ])
    
    # Look-at point (origin)
    look_at = np.array([0, 0, 0])
    
    # Up vector
    up = np.array([0, 1, 0])
    
    # Calculate camera axes
    z_axis = look_at - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    
    # Rotation matrix (world to camera)
    R = np.vstack((x_axis, y_axis, z_axis)).T
    
    # Translation vector
    t = -R @ cam_pos.reshape(3, 1)
    
    return R, t


def draw_3d_point_cloud(points, title="3D Point Cloud", point_size=1, color='b'):
    """
    Visualize 3D points using Matplotlib.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, c=color, alpha=0.8)
    
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def triangulate_pair(K, proj1, proj2, pts1, pts2):
    """
    Triangulate points between two views given projection matrices and their corresponding 2D points.
    Returns points in Euclidean coordinates.
    """
    pts1_ud = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
    pts2_ud = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)
    pts_4d = cv2.triangulatePoints(proj1, proj2, pts1_ud, pts2_ud)
    pts_3d = (pts_4d / pts_4d[3])[:3].T  # convert homogeneous coordinates to 3D
    return pts_3d

def filter_point_cloud(points, eps=1.0, min_samples=10):
    """
    Filter the point cloud using DBSCAN clustering to remove outliers.
    """
    scaler = StandardScaler()
    scaled_points = scaler.fit_transform(points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_points)
    labels = db.labels_
    unique_labels = set(labels)
    largest_cluster_size = 0
    largest_cluster_label = None
    for label in unique_labels:
        if label != -1:  # Exclude noise points
            cluster_size = np.sum(labels == label)
            if cluster_size > largest_cluster_size:
                largest_cluster_size = cluster_size
                largest_cluster_label = label
    if largest_cluster_label is None:
        print("No valid clusters found. Returning original point cloud.")
        return points
    mask = (labels == largest_cluster_label)
    filtered_points = points[mask]
    print(f"Filtered point cloud: {filtered_points.shape[0]} points (from original {points.shape[0]} points)")
    return filtered_points

def interpolate_points(points, target_count=5000):
    """
    Interpolate additional points to create a denser point cloud.
    Uses Delaunay triangulation to create a surface and then samples points on that surface.
    """
    if len(points) < 4:
        print("Not enough points for interpolation")
        return points
    
    try:
        tri = Delaunay(points)
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        random_points = np.random.uniform(low=min_bounds, high=max_bounds, size=(target_count, 3))
        tetra_indices = tri.find_simplex(random_points)
        valid_points = random_points[tetra_indices >= 0]
        combined_points = np.vstack([points, valid_points])
        print(f"Interpolated point cloud: {combined_points.shape[0]} points")
        return combined_points
    except Exception as e:
        print(f"Interpolation failed: {e}")
        return points

def marching_cubes_from_voxels(occupancy, Xs, Ys, Zs):
    """
    Convert an occupancy grid to a surface mesh using marching cubes.
    Returns vertices and faces.
    """
    volume = occupancy.astype(np.uint8)
    vol_reorder = np.transpose(volume, (2, 1, 0))
    dx = Xs[1] - Xs[0]
    dy = Ys[1] - Ys[0]
    dz = Zs[1] - Zs[0]
    verts, faces, normals, values = measure.marching_cubes(vol_reorder, level=0.5, spacing=(dz, dy, dx))
    x0, y0, z0 = Xs[0], Ys[0], Zs[0]
    verts[:, 0] += z0
    verts[:, 1] += y0
    verts[:, 2] += x0
    return verts, faces
