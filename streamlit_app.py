# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import glob
from sfm_pipeline import run_sfm_pipeline

st.title("COIL-100 Structure-from-Motion Viewer")
st.write("Select an object and run the SfM pipeline to view its 3D reconstruction.")

object_choice = st.selectbox("Select Object", ["Object 21", "Object 31"])

if object_choice == "Object 21":
    coil_pattern = "object21/obj21__*.png"
else:
    coil_pattern = "object31/obj31__*.png"

sample_paths = sorted(glob.glob(coil_pattern))
if sample_paths:
    sample_img = cv2.imread(sample_paths[0])
    if sample_img is not None:
        st.image(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB),
                 caption=f"Sample image from {object_choice}",
                 use_column_width=True)
else:
    st.write("No sample images found.")

if st.button("Run SfM Pipeline"):
    st.write("Running SfM pipeline. This may take several minutes…")
    point_cloud, global_R, global_t, used_frame_indices = run_sfm_pipeline(
        coil_pattern,
        use_orb=True,
        interpolate_flag=True,
        filter_flag=True
    )
    st.write(f"SfM pipeline completed. Total 3D points reconstructed: {point_cloud.shape[0]}")
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=2, c="r", alpha=0.8)
    ax.set_title("3D Reconstruction")
    st.pyplot(fig)

    buffer = io.BytesIO()
    np.save(buffer, point_cloud)
    buffer.seek(0)
    st.download_button(
        label="Download Point Cloud (.npy)",
        data=buffer,
        file_name="point_cloud.npy",
        mime="application/octet-stream"
    )
