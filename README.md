# Structure from Motion (SfM) on COIL-100 Dataset  
##Overview  
This repository contains an implementation of a Structure from Motion (SfM) pipeline for 3D reconstruction of objects from the COIL-100 dataset. The pipeline combines classical computer vision techniques with knowledge of the turntable geometry to create sparse 3D point clouds from multiple 2D views of an object.

![3D Reconstruction Example](https://appapppy-jzqc97qqnggkizptx8p6nm.streamlit.app/)  
  
Camera Pose Estimation: Initializes camera poses using known turntable geometry and refines them through feature matching

Multi-Feature Detection: Uses ORB (default) or SIFT feature detectors for keypoint identification

Robust Feature Matching: Implements ratio test and RANSAC for reliable point correspondences

Incremental Reconstruction: Builds the 3D model incrementally by adding new frames to the reconstruction

Post-Processing: Includes DBSCAN clustering for outlier removal and Delaunay-based interpolation for point cloud densification

Requirements
Python 3.7+

OpenCV

NumPy

Matplotlib

SciPy

scikit-learn

scikit-image

Install dependencies:

pip install opencv-python numpy matplotlib scipy scikit-learn scikit-image

Dataset
This implementation is designed for the COIL-100 dataset, which contains 72 images of each object taken at 5-degree rotation intervals on a turntable. Download the dataset from Columbia University's COIL-100 page.

Usage
Place the COIL-100 dataset in a directory named coil-100/coil-100/

Run the main script:

python main.py

By default, the script processes object 21 (wooden lemon squeezer). To process a different object, modify the coil_pattern variable in main.py.

Code Structure
helper_functions.py: Contains utility functions for image processing, camera pose estimation, point cloud manipulation, and visualization

sfm_pipeline.py: Implements the main SfM algorithm

main.py: Entry point that runs the pipeline on a specific object

Methodology
Image Preprocessing: Applies background segmentation, contrast enhancement, and sharpening

Feature Detection: Detects keypoints and computes descriptors using ORB or SIFT

Camera Pose Initialization: Calculates initial camera poses based on the turntable geometry

Feature Matching: Matches features between image pairs using a ratio test

Geometric Verification: Estimates Fundamental and Essential matrices to filter out incorrect matches

Triangulation: Reconstructs 3D points from matched 2D points across multiple views

Post-Processing: Filters outliers and interpolates additional points to create a denser point cloud

Results
The pipeline produces a sparse 3D point cloud representing the object's structure. Results are visualized using Matplotlib and saved as NumPy arrays for further processing.

Limitations
Works best on objects with distinctive texture and features

Reconstruction quality depends on the object's characteristics and lighting conditions

The current implementation produces sparse point clouds rather than dense meshes

Future Work
Implement bundle adjustment for global optimization

Add dense multi-view stereo for more complete reconstructions

Create mesh representations using surface reconstruction algorithms

Improve feature matching for low-texture objects

License
MIT License

Acknowledgments
Columbia University for the COIL-100 dataset

OpenCV community for computer vision algorithms
