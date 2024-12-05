import gzip
import shutil
import pickle
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import threading



# Path to the file
exp_id  = 'tableware_4_8'
source_path = f'/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/{exp_id}/exps/exp_default/pcd_exp_default.pkl.gz'
print(source_path)

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
bowl_pcds = []  # To store point clouds of objects with class_name 'bowl'

for obj in data['objects']:
    pcd_np = obj['pcd_np']  # 3D point cloud data for the object #shape 8512*3, num of pcd * 3D coordinates
    inst_color = obj.get('inst_color', [0.5, 0.5, 0.5]) 
    class_name = next(iter(obj['class_name'])) if isinstance(obj['class_name'], (set, list)) else obj['class_name']

    # Append points and colors to lists
    points.append(pcd_np)
    colors.append(np.tile(inst_color, (pcd_np.shape[0], 1)))  # Repeat color for each point

    # Store unique class_name and color for the legend
    if class_name not in legend_info:
        legend_info[class_name] = inst_color

    # If class_name is 'bowl', store its pcd_np for later
    if class_name == 'cup':
        bowl_pcds.append(pcd_np)

# Combine all points and colors into single arrays
points = np.vstack(points)
colors = np.vstack(colors)

# Set points and colors for Open3D point cloud
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Non-blocking Open3D visualization
def display_point_cloud():
    def empty_callback(vis):
        return False  # Return False to keep the visualization open
    o3d.visualization.draw_geometries_with_animation_callback([point_cloud], empty_callback)

# Display the Matplotlib legend on the main thread
def display_legend():
    fig, ax = plt.subplots(figsize=(4, len(legend_info) * 0.5))  # Adjust size based on number of items
    ax.axis('off')  # Turn off the axis

    # Add a legend entry for each class
    for i, (class_name, color) in enumerate(legend_info.items()):
        # Draw a color bar as a horizontal rectangle
        ax.add_patch(mpatches.Rectangle((0, 0.1*i), 2, 0.1, color=color, transform=ax.transAxes, clip_on=False))
        # Place the class_name next to the color bar, using the color for the text
        ax.text(0.0, 0.1*i + 0.05, f"{class_name}", va='center', ha='left', transform=ax.transAxes, fontsize=10, color="black")

    # Adjust plot limits and show the legend figure
    ax.set_ylim(0, len(legend_info))
    ax.set_xlim(0, 6)
    plt.show()


# Display the point cloud in a non-blocking way
point_cloud_thread = threading.Thread(target=display_point_cloud)
point_cloud_thread.start()

# Display the Matplotlib legend on the main thread
display_legend()

# Wait for the Open3D thread to finish
point_cloud_thread.join()

# Save the point cloud to a .ply file
output_ply_path = os.path.splitext(file_path)[0] + ".ply"
o3d.io.write_point_cloud(output_ply_path, point_cloud)
print(f"Point cloud has been saved to {output_ply_path}")

# Save point clouds of 'bowl' objects to a separate .ply file and visualize them
if bowl_pcds:
    combined_bowl_pcd = np.vstack(bowl_pcds)  # Combine all 'bowl' point clouds
    bowl_point_cloud = o3d.geometry.PointCloud()
    bowl_point_cloud.points = o3d.utility.Vector3dVector(combined_bowl_pcd)
    
    bowl_ply_path = os.path.join(os.path.dirname(file_path), f"{exp_id}_bowl.ply")
    o3d.io.write_point_cloud(bowl_ply_path, bowl_point_cloud)
    print(f"'Bowl' point cloud has been saved to {bowl_ply_path}")

    # Visualize bowl point cloud
    def display_bowl_point_cloud():
        o3d.visualization.draw_geometries([bowl_point_cloud])
    
    bowl_point_cloud_thread = threading.Thread(target=display_bowl_point_cloud)
    bowl_point_cloud_thread.start()
    bowl_point_cloud_thread.join()
else:
    print("No objects with class_name 'bowl' were found.")