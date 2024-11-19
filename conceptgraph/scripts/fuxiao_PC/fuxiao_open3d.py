import gzip
import shutil
import pickle
import numpy as np
import open3d as o3d
import os
# Path to the file
#file_path = '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/conceptgraphs_short/exps/r_mapping_stride2_short/pcd_r_mapping_stride2_short.pkl'
source_path = '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/conceptgraphs_short/exps/mapping_ex1/pcd_mapping_ex1.pkl.gz'
file_path = os.path.splitext(source_path)[0]

with gzip.open(source_path, 'rb') as f_in:
    with open(file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load data from the .pkl file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Initialize an empty Open3D point cloud
point_cloud = o3d.geometry.PointCloud()

# Collect points and colors from each object
points = []
colors = []

for obj in data['objects']:
    pcd_np = obj['pcd_np']  # 3D point cloud data for the object
    inst_color = obj.get('inst_color', [0.5, 0.5, 0.5])  # Default color if not specified
    
    # Append points and colors to lists
    points.append(pcd_np)
    colors.append(np.tile(inst_color, (pcd_np.shape[0], 1)))  # Repeat color for each point

# Combine all points and colors into single arrays
points = np.vstack(points)
colors = np.vstack(colors)

# Set points and colors for Open3D point cloud
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])

# Save the point cloud to a .ply file
output_ply_path = os.path.splitext(file_path)[0] + ".ply"
o3d.io.write_point_cloud(output_ply_path, point_cloud)

print(f"Point cloud has been saved to {output_ply_path}")
