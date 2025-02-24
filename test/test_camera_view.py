import open3d as o3d
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

FOLDER = '/home/hornylemur/repos/concept-graphs/conceptgraph/dataset/external/20250211_172441_scene/exps/r_mapping_stride2_short'
FILE = 'pcd_r_mapping_stride2_short.pkl'

with open(f'{FOLDER}/{FILE}', 'rb') as f:
    data = pickle.load(f)

objects = data['objects']

points = objects[0]['pcd_np']
colors = objects[0]['pcd_color_np']
for i in range(1, len(objects)):
    points = np.concatenate((points, objects[i]['pcd_np']))
    colors = np.concatenate((colors, objects[i]['pcd_color_np']))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)


origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

camera_pose = np.array([[    0.70663,    -0.40553,     0.57985,     0.10939],
                        [   -0.70725,    -0.37952,     0.59646,    0.066181],
                        [  -0.021819,    -0.83157,    -0.55499,      1.3036],
                        [          0,           0,           0,           1]])
camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
camera_frame.transform(camera_pose)

points = np.asarray(pcd.points)
points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
points = points.T
points = np.dot(np.linalg.inv(camera_pose), points)
points = points.T
points = points[:, :3]
pcd.points = o3d.utility.Vector3dVector(points)

'------ Good above here ------'
min_depth = 0.5
max_depth = 3.0
img_width = 240 # 424
img_height = 424 # 240
depth_scale = 1000

intrinsics = np.array([[303.178, 0, 213.676],
                       [0, 303.207, 125.467],
                       [0, 0, 1]])
uv = (intrinsics @ points.T).T
uv[:, 0] /= uv[:, 2]
uv[:, 1] /= uv[:, 2]

total_points = uv.shape[0]
print(uv.shape)
valid_indices = (uv[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < img_width) & \
            (uv[:, 1] >= 0) & (uv[:, 1] < img_height) & \
            (uv[:, 2] >= min_depth) & (uv[:, 2] <= max_depth)
depth_image = np.zeros((img_height, img_width), dtype=np.uint16)
uv_valid = uv[valid_indices]
depth_valid = uv_valid[:, 2] * depth_scale  # Scale depth to millimeters
x_valid = uv_valid[:, 0].astype(np.int32)
y_valid = uv_valid[:, 1].astype(np.int32)
depth_image[y_valid, x_valid] = depth_valid.astype(np.uint16)

plt.imshow(depth_image)
plt.show()
# cv2.imshow('original_image', depth_image)
# cv2.waitKey()

# Get first pose of camera from ros bag
# Get a random pose that definitely does not have the objects
# Test out both my methods of checking objects
# --- camera plane method
# --- projection method
# Debug until it works
# Put back in cg and continue debugging rest of Joe code

geometry = [origin, pcd, camera_frame]
o3d.visualization.draw_geometries(geometry) 

#### LESSONS
## 1. Give a camera pose in world frame, to move points from world frame to camera frame, pre-multiply by the INVERSE of the camera pose.
## 2. Be aware that we are flipping the images to feed into the algorithm. This can have consequences down the line (such as here)




    