import open3d as o3d
import os

# Path to the bowl.ply file
exp_id = 'tableware_5_12'  # Change this if your exp_id is different
file_directory = '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/'  # Base directory
bowl_ply_path = os.path.join(file_directory, exp_id, 'exps', 'exp_default', f"{exp_id}_bowl.ply")

# Check if the file exists
if not os.path.exists(bowl_ply_path):
    raise FileNotFoundError(f"File not found: {bowl_ply_path}")

# Load the point cloud from the .ply file
bowl_point_cloud = o3d.io.read_point_cloud(bowl_ply_path)

print(f"Point cloud has {len(bowl_point_cloud.points)} points.")

# Visualize the point cloud
def visualize_point_cloud(pcd, title="Point Cloud"):
    o3d.visualization.draw_geometries([pcd], window_name=title)

# Denoising using Statistical Outlier Removal
def denoise_point_cloud_sor(pcd, nb_neighbors=20, std_ratio=2):
    """
    Denoise the point cloud using Statistical Outlier Removal.

    :param pcd: Input point cloud (o3d.geometry.PointCloud)
    :param nb_neighbors: Number of neighbors to analyze for each point
    :param std_ratio: Threshold for considering a point an outlier
    :return: Denoised point cloud
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

# Downsampling the point cloud
def downsample_point_cloud(pcd, voxel_size=0.01):
    """
    Downsample the point cloud using voxel grid filtering.

    :param pcd: Input point cloud (o3d.geometry.PointCloud)
    :param voxel_size: Size of the voxel grid
    :return: Downsampled point cloud
    """
    return pcd.voxel_down_sample(voxel_size=voxel_size)






# Apply denoising
denoised_bowl = denoise_point_cloud_sor(bowl_point_cloud, nb_neighbors=400, std_ratio=0.009)
print(f"Denoised Point cloud has {len(denoised_bowl.points)} points.")
# Save the denoised point cloud (optional)
denoised_ply_path = os.path.join(file_directory, exp_id, 'exps', 'exp_default', f"{exp_id}_bowl_denoised.ply")
o3d.io.write_point_cloud(denoised_ply_path, denoised_bowl)
#print(f"Denoised point cloud saved to: {denoised_ply_path}")


# Visualize the original and denoised point clouds sequentially
#print("Visualizing the original point cloud...")
#visualize_point_cloud(bowl_point_cloud, title="Original Bowl Point Cloud")
#print("Visualizing the denoised point cloud...")
visualize_point_cloud(denoised_bowl, title="Denoised Bowl Point Cloud")



# # Downsample the original point cloud
# downsampled_bowl = downsample_point_cloud(bowl_point_cloud, voxel_size=0.02)
# print(f"Downsampled point cloud has {len(downsampled_bowl.points)} points (original: {len(bowl_point_cloud.points)} points).")

# # Visualize the downsampled point cloud
# visualize_point_cloud(downsampled_bowl, title="Downsampled Bowl Point Cloud")
# # Optional: Apply denoising to the downsampled point cloud
# denoised_downsampled_bowl = denoise_point_cloud_sor(downsampled_bowl, nb_neighbors=30, std_ratio=2.0)
# # Visualize the denoised downsampled point cloud
# visualize_point_cloud(denoised_downsampled_bowl, title="Denoised Downsampled Bowl Point Cloud")