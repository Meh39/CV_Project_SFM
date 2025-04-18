import cv2
import matplotlib.pyplot as plt
import numpy as np
from sfm_pipeline import run_sfm_pipeline

sample_img_path = "object21/obj21__10.png"
sample_img = cv2.imread(sample_img_path)
if sample_img is not None:
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Sample: {sample_img_path}")
    plt.axis('off')
    plt.show()
else:
    print("Sample image not found!")

coil_pattern = "object21/obj21__*.png"

point_cloud, global_R, global_t, used_frame_indices = run_sfm_pipeline(
    coil_pattern,
    use_orb=True,
    interpolate_flag=True,
    filter_flag=True
)

np.save("point_cloud.npy", point_cloud)
np.save("camera_rotations.npy", np.array(global_R))
np.save("camera_translations.npy", np.array(global_t))
np.save("used_frame_indices.npy", np.array(used_frame_indices))

print("SfM pipeline completed and results saved.")
