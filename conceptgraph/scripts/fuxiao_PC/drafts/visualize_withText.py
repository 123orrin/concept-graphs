import gzip
import shutil
import pickle
import numpy as np
import open3d as o3d
import os

# Path to the file
source_path = '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/tableware_4_5/exps/exp_default/pcd_exp_default.pkl.gz'
file_path = os.path.splitext(source_path)[0]

# Decompress the .gz file
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
legend_info = {}  # Dictionary to store class_name and color

for obj in data['objects']:
    pcd_np = obj['pcd_np']  # 3D point cloud data for the object
    inst_color = obj.get('inst_color', [0.5, 0.5, 0.5])  # Default color if not specified
    class_name = next(iter(obj['class_name'])) if isinstance(obj['class_name'], (set, list)) else obj['class_name']

    # Append points and colors to lists
    points.append(pcd_np)
    colors.append(np.tile(inst_color, (pcd_np.shape[0], 1)))  # Repeat color for each point

    # Store unique class_name and color for the legend
    if class_name not in legend_info:
        legend_info[class_name] = inst_color

# Combine all points and colors into single arrays
points = np.vstack(points)
colors = np.vstack(colors)

# Set points and colors for Open3D point cloud
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Add text legend as 3D annotations
def add_text_legend(legend_info, point_cloud):
    text_objects = []  # To store text geometries
    offset = 0.2  # Distance between each text annotation
    start_pos = np.min(np.asarray(point_cloud.points), axis=0)  # Start position for text annotations
    start_pos[0] -= 1  # Offset to place text to the left of the point cloud
    
    for idx, (class_name, color) in enumerate(legend_info.items()):
        # Calculate text position
        text_position = start_pos + [0, 0, idx * offset]
        
        # Create a Text3D object
        text = o3d.geometry.Text3D(
            text=class_name,
            position=text_position,
            font_size=20,
            color=color
        )
        text_objects.append(text)
    return text_objects

# Get text legend objects
text_objects = add_text_legend(legend_info, point_cloud)

# Visualize the point cloud and 3D text legend
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the point cloud to the visualizer
vis.add_geometry(point_cloud)

# Add text legend objects to the visualizer
for text in text_objects:
    vis.add_geometry(text)

# Run visualization
vis.run()
vis.destroy_window()

# Save the point cloud to a .ply file
output_ply_path = os.path.splitext(file_path)[0] + ".ply"
o3d.io.write_point_cloud(output_ply_path, point_cloud)
print(f"Point cloud has been saved to {output_ply_path}")
