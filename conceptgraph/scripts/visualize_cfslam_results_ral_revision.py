import cv2
import os
import gc
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
from pathlib import Path
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
import open_clip

import distinctipy

# from conceptgraph.utils.pointclouds import Pointclouds
from conceptgraph.utils.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh
from conceptgraph.slam.utils import filter_objects, merge_objects

from conceptgraph.scripts.gpt_object_classes import object_classes as ext_scannet_classes
from conceptgraph.scripts.llm_prompting import required_semantic_safety_constraints
from conceptgraph.scripts.prompts import semantic_types, constraint_types
from conceptgraph.scripts.clip_text_cache import CLIPTextCache


def interpolate_missing_properties(df_source, df_query, k_nearest=3):
    import pandas as pd
    from scipy.spatial import KDTree
    xyz = list('xyz')

    print('generating a simplified point cloud (this may take a while...)')

    tree = KDTree(df_source[xyz].values)
    _, ii = tree.query(df_query[xyz], k=k_nearest)
    n = df_query.shape[0]

    df_result = pd.DataFrame(0, index=range(n), columns=df_source.columns)
    df_result[xyz] = df_query[xyz]
    other_cols = [c for c in df_source.columns if c not in xyz]

    for i in range(n):
        m = df_source.loc[ii[i].tolist(), other_cols].mean(axis=0)
        df_result.loc[i, other_cols] = m

    return df_result

def exclude_points(df_source, df_exclude, radius):
    from scipy.spatial import KDTree
    xyz = list('xyz')
    tree = KDTree(df_exclude[xyz].values)
    ii = tree.query_ball_point(df_source[xyz], r=radius, return_length=True)
    mask = [l == 0 for l in ii]
    df_result = df_source.iloc[mask]
    return df_result

def voxel_decimate(df, cell_size):
    def grouping_function(row):
        return tuple([round(row[c] / cell_size) for c in 'xyz'])
    grouped = df.assign(voxel_index=df.apply(grouping_function, axis=1)).groupby('voxel_index')
    return grouped.first().reset_index()[[c for c in df.columns if c != 'voxel_index']]

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

    parser.add_argument("--ee_object", type=str, default=None)
    
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
    detected_classes = [obj["class_name"] for obj in results["objects"]]
    detected_classes.sort()
    print(detected_classes)
    objects.load_serializable(results["objects"])
    bg_objects = MapObjectList()
    bg_objects.extend(obj for obj in objects if obj['is_background'])
    if len(bg_objects) == 0:
        bg_objects = None
    class_colors = results['class_colors']
        
    return objects, bg_objects, class_colors


# Function to convert camera parameters to a dictionary
def convert_camera_params_to_dict(param):
    return {
        "class_name": param.__class__.__name__,
        "intrinsic": {
            "width": param.intrinsic.width,
            "height": param.intrinsic.height,
            "intrinsic_matrix": param.intrinsic.intrinsic_matrix.tolist()
        },
        "extrinsic": param.extrinsic.tolist()
    }


def main(ral_revision, args, debug_transform=False):
    result_path = args.result_path
    rgb_pcd_path = args.rgb_pcd_path
    
    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."

    if rgb_pcd_path is not None:        
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)

        # entities.append(global_pcd)
        # o3d.visualization.draw_geometries(entities)
        # exit()
        
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

    # o3d.visualization.draw_geometries(pcds + bboxes)
        
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
        
    def color_by_clip_sim(vis, query=None, highlight_max=False):
        if args.no_clip:
            print("CLIP model is not initialized.")
            return

        if query is None:
            text_query = input("Enter your query: ")
        else:
            text_query = query
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

        if query:
            return max_prob_idx, pcds[max_prob_idx], max_prob_object['bbox']

    def save_view_params(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("temp.json", param)

    def encode_text_batches(text_queries, clip_model, clip_tokenizer, batch_size=16, device="cuda"):
        """
        Memory-efficient batch encoding of text queries using CLIP.
        
        Args:
            text_queries: List of text queries to encode
            clip_model: CLIP model instance
            clip_tokenizer: CLIP tokenizer instance
            batch_size: Number of queries to process at once
            device: Device to run encoding on ("cuda" or "cpu")
        
        Returns:
            torch.Tensor: Concatenated text features
        """
        text_query_fts = []
        num_batches = (len(text_queries) + batch_size - 1) // batch_size

        for batch_id, i in enumerate(range(0, len(text_queries), batch_size)):
            print(f"Processing batch {batch_id + 1} of {num_batches}")
            
            # Move model to CPU temporarily if memory is critical
            # clip_model.to("cpu")
            # torch.cuda.empty_cache()
            # clip_model.to(device)
            
            batch_queries = text_queries[i:i + batch_size]
            
            with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
                # Tokenize and encode in the same context
                text_queries_tokenized = clip_tokenizer(batch_queries).to(device)
                text_query_ft = clip_model.encode_text(text_queries_tokenized)
                
                # Convert to CPU and append
                text_query_fts.append(text_query_ft.cpu())
                
                # Explicit cleanup
                del text_queries_tokenized
                del text_query_ft
                torch.cuda.empty_cache()
                gc.collect()
        
        # Concatenate all features on CPU to avoid GPU memory issues
        final_features = torch.cat(text_query_fts, dim=0)
        
        return final_features
    
    def get_text_embeddings(text_queries, clip_model, clip_tokenizer, batch_size=32, cache_dir="./clip_cache"):
        """Main function to get text embeddings with caching."""
        # Initialize cache
        cache = CLIPTextCache(cache_dir)
        
        # Try to get cached embeddings
        cached_embeddings = cache.get_cached_embeddings(text_queries)
        if cached_embeddings is not None:
            return cached_embeddings.to("cuda")
        
        # If not cached, compute embeddings
        text_query_fts = encode_text_batches(text_queries, clip_model, clip_tokenizer, batch_size=batch_size)
        text_query_fts = text_query_fts / text_query_fts.norm(dim=-1, keepdim=True)
        
        # Cache the results before moving to GPU
        cache.cache_embeddings(text_queries, text_query_fts.cpu())
        
        # Return GPU tensor
        return text_query_fts.to("cuda")

    def classify(vis, batch_size=16, top_n=5, debug=False):
        if args.no_clip:
            print("CLIP model is not initialized.")
            return

        text_queries = ext_scannet_classes

        # encode the text queries in batches
        text_query_fts = get_text_embeddings(text_queries, clip_model, clip_tokenizer, 
                                             batch_size=batch_size, 
                                             cache_dir="conceptgraph/dataset/clip_cache")
        
        # similarities = objects.compute_similarities(text_query_ft)
        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        
        print()
        num_objects = len(objects)
        object_classes_dict = {}
        for object_id in range(num_objects):
            similarities = F.cosine_similarity(text_query_fts, objects_clip_fts[object_id, :].unsqueeze(0), dim=-1)
            max_value = similarities.max()
            min_value = similarities.min()
            normalized_similarities = (similarities - min_value) / (max_value - min_value)
            probs = F.softmax(similarities, dim=0)
            
            top_n_probs, top_n_indices = torch.topk(probs, top_n)

            object_classes_dict[object_id] = {
                "class_name": objects[object_id]['class_name'],
                "top_n_probs": top_n_probs,
                "top_n_indices": top_n_indices
            }
            object_classes_dict[object_id]["top_n_classes"] = [] 
            # Print the top n classes
            print(f"Object {object_id} with class name '{objects[object_id]['class_name']}'")
            for i in range(top_n):
                class_name = text_queries[top_n_indices[i]]
                print(f"Top {i + 1} class: '{class_name}' with probability {top_n_probs[i].item()}")
                object_classes_dict[object_id][f"top_n_classes"].append((i, class_name))
            print()

            if debug:
                # Highlight the object of interest in red
                pcd = pcds[object_id]
                map_colors = np.asarray(pcd.colors)
                pcd.colors = o3d.utility.Vector3dVector(
                    np.tile([1.0, 0.0, 0.0], (len(pcd.points), 1))
                )

                # Turn the other objects to gray
                for i in range(num_objects):
                    if i == object_id:
                        continue
                    pcd = pcds[i]
                    pcd.colors = o3d.utility.Vector3dVector(
                        np.tile([0.5, 0.5, 0.5], (len(pcd.points), 1))
                    )

                o3d.visualization.draw_geometries(pcds)
        
        return object_classes_dict

    def find_desk(vis, query="desk", debug=True):
        max_prob_idx, max_pcd, max_bbox = color_by_clip_sim(vis, query=query)
        # print("Most probable object is at index", max_prob_idx)
        # print("max pcd:", max_pcd)
        # print("max bbox:", max_bbox)
        max_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        max_pcd.orient_normals_consistent_tangent_plane(k=15)

        assert (max_pcd.has_normals())

        # # using all defaults
        # oboxes = max_pcd.detect_planar_patches(
        #     normal_variance_threshold_deg=60,
        #     coplanarity_deg=75,
        #     outlier_ratio=0.75,
        #     min_plane_edge_length=0,
        #     min_num_points=0,
        #     search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        
        # using all defaults
        oboxes = max_pcd.detect_planar_patches(
            normal_variance_threshold_deg=60,
            coplanarity_deg=75,
            outlier_ratio=0.1,
            min_plane_edge_length=0,
            min_num_points=0,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        print("Detected {} patches".format(len(oboxes)))

        assert (len(oboxes) >= 3) # we need at least 3 planes to form a box

        geometries = []
        entities = [max_pcd]
        planes = []
        if debug:
            for obox in oboxes:
                mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
                mesh.paint_uniform_color(obox.color)

                print("Color: ", obox.color)

                vis.add_geometry(mesh)
                planes.append(mesh)

        return oboxes, planes

    def find_robot(vis, query="robot"):
        max_prob_idx, max_pcd, max_bbox = color_by_clip_sim(vis, query=query)

        # labels = np.array(max_pcd.cluster_dbscan(eps=0.15, min_points=10, print_progress=True))
        labels = np.array(max_pcd.cluster_dbscan(eps=0.15, min_points=10, print_progress=True))
        max_label = labels.max()
        print(max_pcd)
        print(f"point cloud has {max_label + 1} clusters")

        # remove negative labels
        labels = labels[labels >= 0]
        # Find the label that occurs the most
        max_label = np.argmax(np.bincount(labels))
        print("max label: ", max_label)

        max_pcd_indices = np.where(labels == max_label)[0]
        max_pcd_points = np.asarray(max_pcd.points)[max_pcd_indices]
        max_pcd_colors = np.asarray(max_pcd.colors)[max_pcd_indices]
        max_pcd = o3d.geometry.PointCloud()
        max_pcd.points = o3d.utility.Vector3dVector(max_pcd_points)
        max_pcd.colors = o3d.utility.Vector3dVector(max_pcd_colors)

        max_pcd, _ = max_pcd.remove_radius_outlier(nb_points=10, radius=0.03)

        return max_pcd

    def get_robot_pcd_from_classification(object_classes_dict):
        robot_queries = ["robot", "robot arm"]
        robot_ids = []

        for object_id in object_classes_dict:
            obj = object_classes_dict[object_id]
            top_n_classes = obj["top_n_classes"]
            for i, class_name in top_n_classes:
                # Only consider the first class
                if i > 0:
                    break

                if class_name in robot_queries:
                    robot_ids.append(object_id)
                    break

        if len(robot_ids) == 0:
            print("No robot found.")
            return None

        robot_pcd = o3d.geometry.PointCloud()
        for robot_id in robot_ids:
            pcd = objects[robot_id]["pcd"]
            robot_pcd += pcd

        return robot_pcd

    def get_robot_pcd(vis, object_classes_dict):
        robot_pcd = get_robot_pcd_from_classification(object_classes_dict)
        if robot_pcd is None:
            robot_pcd = find_robot(vis, query="robot")

        return robot_pcd

    def find_closest_orthogonal_basis(normals):
        """
        Find the three plane normals that are closest to forming an orthogonal basis.
        
        Parameters:
        normals: numpy array of shape (n, 3) where n is the number of plane normals
        
        Returns:
        best_basis: indices of the three normals that best form an orthogonal basis
        orthogonality_score: measure of how close to orthogonal (lower is better)
        """
        # Normalize all vectors
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        
        # Get all possible combinations of three normals
        n = len(normals)
        combinations_of_three = list(combinations(range(n), 3))
        
        best_score = float('inf')
        best_basis = None
        
        for combo in combinations_of_three:
            # Get the three vectors
            v1, v2, v3 = normals[list(combo)]
            
            # Calculate dot products between all pairs
            dot12 = np.abs(np.dot(v1, v2))
            dot23 = np.abs(np.dot(v2, v3))
            dot13 = np.abs(np.dot(v1, v3))
            
            # For perfectly orthogonal vectors, all dot products would be 0
            # Sum the absolute values of dot products as our score
            score = dot12 + dot23 + dot13
            
            if score < best_score:
                best_score = score
                best_basis = combo
                
        return best_basis, best_score

    def identify_robot_transformation(vis, object_classes_dict, debug=False, y_axis_first=False):
        print("Identifying desk")
        # First step: Find the desk and its planes
        oboxes, _ = find_desk(vis, debug=False)

        print("Getting robot pointcloud")
        # Second step: Find the robot
        robot_pcd = get_robot_pcd(vis, object_classes_dict)

        # Third step: Determine the distance of all robot points to the desk planes
        distances = []
        thresholds = []
        planes = []
        for obox in oboxes:
            # Get the normal of the plane
            normal = obox.R @ np.array([0, 0, 1])
            center = obox.center
            bias = -np.dot(normal, center)

            planes.append((normal, bias))

            # Calculate the distance of all points to the plane
            dist = np.abs(np.dot(robot_pcd.points, normal) + bias) / np.linalg.norm(normal)
            distances.append(dist)

            percentile = 0.05
            threshold = np.quantile(dist, percentile)
            thresholds.append(threshold)

            if debug:
                # plot a histogram of the distances
                plt.hist(dist, bins=10)
                plt.axvline(x=threshold, linestyle='--')            
        
        if debug:
            plt.show()

        # Fourth step: Find the plane that is closest to the bottom of the robot
        # The plane that is closest to the bottom of the robot is the table top
        table_top_plane_id = np.argmin(thresholds)
        table_top_plane = planes[table_top_plane_id]
        table_top_bbox = oboxes[table_top_plane_id]

        if len(planes) > 3:
            print("More than 3 planes detected. Selecting the 3 planes that are most orthogonal.")
            normals = np.array([plane[0] for plane in planes])
            best_indices, score = find_closest_orthogonal_basis(normals)

            print(f"Best basis indices: {best_indices}")
            print(f"Orthogonality score (lower is better): {score:.6f}")
            print("\nSelected vectors:")
            for idx in best_indices:
                print(f"Vector {idx}: {normals[idx]}")

            assert table_top_plane_id in best_indices # The table top plane should be one of the selected planes
        
        # Determine the transformation matrix for the table top
        T_table_top = np.eye(4)
        T_table_top[:3, :3] = table_top_bbox.R
        T_table_top[:3, 3] = table_top_bbox.center

        print("Table top plane is at index", table_top_plane_id)
        print("Table top plane: ", table_top_plane)

        if debug:
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(table_top_bbox, scale=[1, 1, 0.0001])
            mesh.paint_uniform_color(obox.color)

            vis.add_geometry(mesh)
            planes.append(mesh)

            # Add a coordinate frame at the center of the bounding box
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            frame.transform(T_table_top)
            vis.add_geometry(frame)        

            # Color the robot point cloud by the distance to the table top plane
            max_dist = np.max(distances[table_top_plane_id])
            colors = np.zeros((len(robot_pcd.points), 3))
            for i, dist in enumerate(distances[table_top_plane_id]):
                colors[i] = cmap(dist / max_dist)[:3]

            robot_pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(robot_pcd)

        # Fifth step: Find the bottom of the robot
        # The points that are closer than the threshold to the table top plane are the bottom of the robot and define the center of the robot base
        # bottom_points = np.asarray(robot_pcd.points)[np.where(distances[table_top_plane_id] < thresholds[table_top_plane_id])]
        # Using the points that are between 0.15 and 0.25 meters from the table top plane as the bottom of the robot
        # as the base has a larger support area that is not centered at the center of the first joint.  
        bottom_points = np.asarray(robot_pcd.points)[np.where((0.15 <= distances[table_top_plane_id]) & (distances[table_top_plane_id] <= 0.25))]
        bottom_center = np.mean(bottom_points, axis=0)

        # Project the bottom center to the table top plane
        normal = table_top_plane[0]
        bias = table_top_plane[1]
        projected_center = bottom_center - (np.dot(bottom_center, normal) + bias) / np.linalg.norm(normal) * normal

        if debug:
            # Add a sphere at the bottom center
            sphere = create_ball_mesh(bottom_center, 0.05, color=(1, 0, 0))
            vis.add_geometry(sphere)

            # Add a sphere at the projected center
            sphere = create_ball_mesh(projected_center, 0.05, color=(0, 1, 0))
            vis.add_geometry(sphere)

        # Sixth step: Determine the orientation of the robot
        # Check which side of the plane the robot point cloud is on
        # If the robot point cloud is on the negative side of the plane, the normal should be flipped
        robot_pcd_plane_offset = np.dot(robot_pcd.points, normal) + bias

        percentile = 0.05
        threshold = np.quantile(robot_pcd_plane_offset, percentile)
        print("Threshold: ", threshold)

        if debug: 
            # Plot a histogram of the robot point cloud plane offset
            plt.hist(robot_pcd_plane_offset, bins=10)
            plt.axvline(x=threshold, linestyle='--')
            plt.show()    

        if threshold < 0:
            normal = -normal

        z_axis = normal   

        median_distances = {}
        table_side_plane_id = None
        if y_axis_first:
            minimum_median_distance = float('inf')
        else:
            maximum_median_distance = 0.0

        for threshold_id, threshold in enumerate(thresholds):
            print("Threshold: ", threshold)
            if threshold_id == table_top_plane_id:
                print("Skipping table top plane")
                continue

            if not threshold_id in best_indices:
                print("Skipping non-orthogonal plane")
                continue

            dist = distances[threshold_id]

            percentile = 0.5
            threshold = np.quantile(dist, percentile)
            median_distances[threshold_id] = threshold

            if y_axis_first:
                if threshold < minimum_median_distance:
                    minimum_median_distance = threshold
                    table_side_plane_id = threshold_id
            else:
                if threshold > maximum_median_distance:
                    maximum_median_distance = threshold
                    table_side_plane_id = threshold_id

            if debug:
                # plot a histogram of the distances
                plt.hist(dist, bins=10)
                plt.axvline(x=threshold, linestyle='--')
                plt.xlim(left=0.0)
                plt.show()
        
        table_side_plane = planes[table_side_plane_id]

        if debug:
            table_side_bbox = oboxes[table_side_plane_id]

            # Determine the transformation matrix for the table side
            T_table_side = np.eye(4)
            T_table_side[:3, :3] = table_side_bbox.R
            T_table_side[:3, 3] = table_side_bbox.center

            print("Table side plane is at index", table_side_plane_id)
            print("Table side plane: ", table_side_plane)
            
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(table_side_bbox, scale=[1, 1, 0.0001])
            mesh.paint_uniform_color(obox.color)

            vis.add_geometry(mesh)
            planes.append(mesh)

            # Add a coordinate frame at the center of the bounding box
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            frame.transform(T_table_side)
            vis.add_geometry(frame)

        # The considered axis is the normal of the table side plane
        axis = table_side_plane[0]

        if not y_axis_first:
            # Check if more points are on the positive side of the plane 
            offset = np.dot(robot_pcd.points - projected_center, axis)

            if debug:
                print("Offset mean: ", np.mean(offset))
                # Plot a histogram of the offset
                plt.hist(offset, bins=10)
                plt.show()
            
            if np.mean(offset) > 0:
                axis = -axis

        # Project the y axis to the table top plane
        axis = axis - np.dot(axis, z_axis) / np.linalg.norm(normal) * z_axis

        # Check if y_axis is in the plane
        if np.abs(np.dot(axis, z_axis)) > 1e-6:
            print("Axis is not in the plane. Incorrect projection!")

        # The remaining axis is the cross product of the axis and the z axis
        other_axis = np.cross(axis, z_axis)

        if y_axis_first:
            x_axis = other_axis
            y_axis = axis
        else:
            x_axis = axis
            y_axis = -other_axis

        # Create the rotation matrix
        R = np.eye(3)
        R[:, 0] = x_axis
        R[:, 1] = y_axis
        R[:, 2] = z_axis

        # Create the transformation matrix
        T_RO = np.eye(4)  # origin to robot
        T_RO[:3, :3] = R
        T_RO[:3, 3] = projected_center

        if debug:
            # Add a frame at the projected center
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            frame.transform(T_RO)
            vis.add_geometry(frame)

        T_OR = np.linalg.inv(T_RO)  # robot to origin

        return T_OR

    def object_labels(object_classes_dict):
        robot_queries = ["robot", "robot arm"]
        scene_objects = {}

        for object_id in object_classes_dict:
            obj = object_classes_dict[object_id]
            top_n_classes = obj["top_n_classes"]
            for i, class_name in top_n_classes:
                # Only consider the first class
                if i > 0:
                    break

                if not class_name in robot_queries:
                    scene_objects[object_id] = class_name

        print(scene_objects)

        return scene_objects

    if ral_revision:
        debug_classification = False
        object_classes_dict = classify(vis, debug=debug_classification)

        scene_objects = object_labels(object_classes_dict)
        scene_objects_wo_duplicates = list(set(scene_objects.values()))
        print(scene_objects_wo_duplicates)

        if args.ee_object is not None:
            ee_object = args.ee_object
            semantic_safety = required_semantic_safety_constraints([ee_object], 
                                                                   semantic_types, 
                                                                   scene_objects_wo_duplicates, 
                                                                   constraint_types, 
                                                                   repetitions=1, 
                                                                   debug=True)

        T_OR = identify_robot_transformation(vis, object_classes_dict, debug=debug_transform)

        for geometry in pcds:
            geometry.transform(T_OR)

        for bbox in bboxes:
            vis.remove_geometry(bbox)

        # draw a frame at the origin for reference
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(frame)

        # Save the transformed point clouds of the scene to a ply file
        print(os.path.realpath(result_path))
        filename = os.path.realpath(result_path).split(".")[0].split(".")[0]
        for i, pcd in enumerate(pcds):
            o3d.io.write_point_cloud("{}_{}.ply".format(filename, i), pcd)

    else:
        # Color the object based on RGB
        color_by_rgb(vis)

        # Save the transformed point clouds of the scene to a ply file
        print(os.path.realpath(result_path))
        filename = os.path.realpath(result_path).split(".")[0].split(".")[0]
        for i, pcd in enumerate(pcds):
            o3d.io.write_point_cloud("{}_{}.ply".format(filename, i), pcd)

        # color_by_clip_sim(vis, query="fan", highlight_max=False)

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
    ral_revision = True
    debug_transform = False

    parser = get_parser()
    args = parser.parse_args()
    
    main(ral_revision, args, debug_transform=debug_transform)
