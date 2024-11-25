import cv2
import os
# import PyQt5

# # Set the QT_QPA_PLATFORM_PLUGIN_PATH environment variable
# pyqt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt", "plugins", "platforms")
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugin_path

import copy
import json
import os
import pickle
import gzip
import argparse

import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
import open_clip
from pathlib import Path

import distinctipy

# from conceptgraph.utils.pointclouds import Pointclouds
from conceptgraph.utils.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh
from conceptgraph.slam.utils import filter_objects, merge_objects

def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Create a colored mesh sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    
    parser.add_argument("--no_clip", action="store_true", 
                        help="If set, the CLIP model will not init for fast debugging.")
    
    # To inspect the results of merge_overlap_objects
    # This is mainly to quickly try out different thresholds
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    
    return parser

def load_result(result_path):
    # check if theres a potential symlink for result_path and resolve it
    potential_path = os.path.realpath(result_path)
    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        raise ValueError("Results should be a dictionary! other types are not supported!")
    
    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    bg_objects = MapObjectList()
    bg_objects.extend(obj for obj in objects if obj['is_background'])
    if len(bg_objects) == 0:
        bg_objects = None
    class_colors = results['class_colors']
        
    
        
    return objects, bg_objects, class_colors

def main(args):
    result_path = args.result_path
    rgb_pcd_path = args.rgb_pcd_path
    
    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."

    if rgb_pcd_path is not None:        
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)
        
        if result_path is None:
            print("Only visualizing the pointcloud...")
            o3d.visualization.draw_geometries([global_pcd])
            exit()
        
    objects, bg_objects, class_colors = load_result(result_path)
    
    if args.edge_file is not None:
        # Load edge files and create meshes for the scene graph
        scene_graph_geometries = []
        with open(args.edge_file, "r") as f:
            edges = json.load(f)
        
        classes = objects.get_most_common_class()
        colors = [class_colors[str(c)] for c in classes]
        obj_centers = []
        for obj, c in zip(objects, colors):
            pcd = obj['pcd']
            bbox = obj['bbox']
            points = np.asarray(pcd.points)
            center = np.mean(points, axis=0)
            extent = bbox.get_max_bound()
            extent = np.linalg.norm(extent)
            # radius = extent ** 0.5 / 25
            radius = 0.10
            obj_centers.append(center)

            # remove the nodes on the ceiling, for better visualization
            ball = create_ball_mesh(center, radius, c)
            scene_graph_geometries.append(ball)
            
        for edge in edges:
            if edge['object_relation'] == "none of these":
                continue
            id1 = edge["object1"]['id']
            id2 = edge["object2"]['id']

            line_mesh = LineMesh(
                points = np.array([obj_centers[id1], obj_centers[id2]]),
                lines = np.array([[0, 1]]),
                colors = [1, 0, 0],
                radius=0.02
            )

            scene_graph_geometries.extend(line_mesh.cylinder_segments)
    
    if not args.no_clip:
        print("Initializing CLIP model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to("cuda")
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    if bg_objects is not None:
        indices_bg = []
        for obj_idx, obj in enumerate(objects):
            if obj['is_background']:
                indices_bg.append(obj_idx)
        # indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
        # objects.extend(bg_objects)
        
    # Sub-sample the point cloud for better interactive experience
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        # pcd = pcd.voxel_down_sample(0.05)
        objects[i]['pcd'] = pcd
    
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    
    # Get the color for each object when colored by their class
    object_classes = []
    for i in range(len(objects)):
        obj = objects[i]
        pcd = pcds[i]
        obj_classes = np.asarray(obj['class_id'])
        # Get the most common class for this object as the class
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)
    
    # Set the title of the window
    vis = o3d.visualization.VisualizerWithKeyCallback()

    if result_path is not None:
        vis.create_window(window_name=f'Open3D - {os.path.basename(result_path)}', width=1280, height=720)
    else:
        vis.create_window(window_name=f'Open3D', width=1280, height=720)

    # Add geometry to the scene
    for geometry in pcds + bboxes:
        vis.add_geometry(geometry)
        
    main.show_bg_pcd = True
    def toggle_bg_pcd(vis):
        if bg_objects is None:
            print("No background objects found.")
            return
        
        for idx in indices_bg:
            if main.show_bg_pcd:
                vis.remove_geometry(pcds[idx], reset_bounding_box=False)
                vis.remove_geometry(bboxes[idx], reset_bounding_box=False)
            else:
                vis.add_geometry(pcds[idx], reset_bounding_box=False)
                vis.add_geometry(bboxes[idx], reset_bounding_box=False)
        
        main.show_bg_pcd = not main.show_bg_pcd
        
    main.show_global_pcd = False
    def toggle_global_pcd(vis):
        if args.rgb_pcd_path is None:
            print("No RGB pcd path provided.")
            return
        
        if main.show_global_pcd:
            vis.remove_geometry(global_pcd, reset_bounding_box=False)
        else:
            vis.add_geometry(global_pcd, reset_bounding_box=False)
        
        main.show_global_pcd = not main.show_global_pcd
        
    main.show_scene_graph = False
    def toggle_scene_graph(vis):
        if args.edge_file is None:
            print("No edge file provided.")
            return
        
        if main.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
        
        main.show_scene_graph = not main.show_scene_graph
        
    def color_by_class(vis):
        for i in range(len(objects)):
            pcd = pcds[i]
            obj_class = object_classes[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    class_colors[str(obj_class)],
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_rgb(vis):
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = objects[i]['pcd'].colors
        
        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_instance(vis):
        instance_colors = cmap(np.linspace(0, 1, len(pcds)))
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    instance_colors[i, :3],
                    (len(pcd.points), 1)
                )
            )
            
        for pcd in pcds:
            vis.update_geometry(pcd)
        
    def color_by_clip_sim(vis, text_query=None, highlight_max=False):
        if args.no_clip:
            print("CLIP model is not initialized.")
            return

        if text_query is None:
            text_query = input("Enter your query: ")
        text_queries = [text_query]
        
        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        
        # similarities = objects.compute_similarities(text_query_ft)
        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]

        max_prob_object = objects[max_prob_idx]
        print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
        print(f"location xyz: {max_prob_object['bbox'].center}")
        
        if not highlight_max:
            for i in range(len(objects)):
                pcd = pcds[i]
                map_colors = np.asarray(pcd.colors)
                pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(
                        [
                            similarity_colors[i, 0].item(),
                            similarity_colors[i, 1].item(),
                            similarity_colors[i, 2].item()
                        ], 
                        (len(pcd.points), 1)
                    )
                )
        else:
            gray = np.array([0.5, 0.5, 0.5])
            for i in range(len(objects)):
                pcd = pcds[i]
                pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(
                        [
                            gray[0].item(),
                            gray[1].item(),
                            gray[2].item()
                        ], 
                        (len(pcd.points), 1)
                    )
                )

            max_pcd = pcds[max_prob_idx]
            max_pcd.colors = o3d.utility.Vector3dVector(np.tile(
                        [
                            similarity_colors[max_prob_idx, 0].item(),
                            similarity_colors[max_prob_idx, 1].item(),
                            similarity_colors[max_prob_idx, 2].item()
                        ], 
                        (len(max_pcd.points), 1)
                    ))

        for pcd in pcds:
            vis.update_geometry(pcd)

        return max_prob_idx, pcds[max_prob_idx], max_prob_object['bbox']
            
    def save_view_params(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("temp.json", param)

    def find_fan(scene_path, text_query="fan", minimum_vis_points=10):
        max_prob_idx, max_pcd, max_bbox = color_by_clip_sim(vis, text_query=text_query, highlight_max=True)

        # TODO: Deal with multiple fans

        # Calculate the "center of mass" of the fan's point cloud
        points = np.asarray(max_pcd.points)
        center = np.mean(points, axis=0)

        # Draw a sphere at the center of the fan
        radius = 0.01
        mesh_sphere = create_ball_mesh(center, radius, color=(0, 1, 0))
        vis.add_geometry(mesh_sphere)

        # Find the images that contain the fan. Since we know the fan's point cloud and the camera parameters, we can determine if any of the fan's points are visible in the image.
        # import pdb; pdb.set_trace()
        scene_path = str(scene_path)
        rgb_dir = scene_path + "/color"
        poses_dir = scene_path + "/pose"
        intrinsics_file = scene_path + "/intrinsics/000000.npy"

        # Load the camera intrinsics
        intrinsics = np.load(intrinsics_file)
        intrinsics = intrinsics[:3, :3]

        print("intrinsics: ", intrinsics)   

        # Determine the number of images and poses in the scene
        num_images = len(os.listdir(rgb_dir))
        num_poses = len(os.listdir(poses_dir))
        num_valid_poses = min(num_images, num_poses)

        obj_is_visible = [0] * num_valid_poses

        points_inertial_frame = np.hstack((points, np.ones((points.shape[0], 1))))

        num_poses_w_fan = 0

        # Go through each pose and check if a point in the object's point cloud is visible in the image
        for i in range(num_valid_poses):
            # Load the image
            rgb_file = rgb_dir + f"/{i:06d}.png"
            rgb = cv2.imread(rgb_file)

            height, width, _ = rgb.shape

            # Load the pose
            pose_file = poses_dir + f"/{i:06d}.npy"
            pose = np.load(pose_file)

            # Project the object's point cloud into the image using the camera intrinsics and pose
            # Transform the points using the extrinsics
            points_camera_frame = (np.linalg.inv(pose) @ points_inertial_frame.T).T
            points_camera_frame = points_camera_frame[:, :3]

            # Project points onto the 2D image plane using intrinsics
            uv = (intrinsics @ points_camera_frame.T).T
            uv[:, 0] /= uv[:, 2]
            uv[:, 1] /= uv[:, 2]

            # Keep only valid points that are in front of the camera, inside the image frame, and within the depth range
            valid_indices = (uv[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)

            uv_valid = uv[valid_indices, :2]
            x_valid = np.clip(uv_valid[:, 0].astype(np.int32), 0, width - 1)
            y_valid = np.clip(uv_valid[:, 1].astype(np.int32), 0, height - 1)

            rgb[y_valid, x_valid] = [0, 0, 255]

            if len(uv_valid) > minimum_vis_points:
                num_poses_w_fan += 1

                cv2.imshow("visible object", rgb)
                cv2.waitKey(0)

        print(f"Number of poses with the fan visible: {num_poses_w_fan}")

    scene_path = Path(os.path.realpath(result_path)).parents[2]    
    find_fan(scene_path)

    # # Show the pose in the point cloud by adding a coordinate frame at the camera's position
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # frame.transform(pose)
    # vis.add_geometry(frame)

    # # Show the points in the point cloud that are visible in the image in green
    # points_visible_pcd = o3d.geometry.PointCloud()
    # points_visible_pcd.points = o3d.utility.Vector3dVector(valid_points)
    # points_visible_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (len(valid_points), 1)))
    # vis.add_geometry(points_visible_pcd)

    vis.register_key_callback(ord("B"), toggle_bg_pcd)
    vis.register_key_callback(ord("S"), toggle_global_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("F"), color_by_clip_sim)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("V"), save_view_params)
    vis.register_key_callback(ord("G"), toggle_scene_graph)
    
    # Render the scene
    vis.run()
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

