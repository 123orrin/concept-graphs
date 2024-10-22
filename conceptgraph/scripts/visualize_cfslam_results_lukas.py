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

import distinctipy

# from conceptgraph.utils.pointclouds import Pointclouds
from conceptgraph.utils.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh
from conceptgraph.slam.utils import filter_objects, merge_objects


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

def visualize_fr3(vis, fr3_path):
    # Get true robot model for visualization
    robot_pcd = o3d.io.read_point_cloud(fr3_path)

    # Rotate the robot model around the x axis by 90 degrees
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    T = np.eye(4)
    T[:3, :3] = R

    # Rotate the robot model around the z axis by 180 degrees
    R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    T2 = np.eye(4)
    T2[:3, :3] = R

    T = T @ T2
    robot_pcd.transform(T)

    vis.add_geometry(robot_pcd)

    # save the robot pcd to a ply file
    o3d.io.write_point_cloud("fr3_robot_pcd.ply", robot_pcd)

    return robot_pcd


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



# Load the saved camera parameters
def load_camera_params(param_file="camera_params.json"):
    return o3d.io.read_pinhole_camera_parameters(param_file)

# Function to convert camera parameters to the required components
def convert_camera_params(param):
    intrinsic = param.intrinsic
    extrinsic = param.extrinsic

    # The eye is the camera location
    eye = np.linalg.inv(extrinsic)[:3, 3]

    # The lookat point is along the principal axis, which is typically the Z axis in camera coordinates
    lookat = eye + np.linalg.inv(extrinsic)[:3, 2]

    # The up direction is the Y axis in camera coordinates
    up = np.linalg.inv(extrinsic)[:3, 1]

    # The field of view can be computed from the intrinsic parameters
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    width = intrinsic.width
    height = intrinsic.height

    # Calculate the field of view (horizontal)
    fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi

    return {
        "lookat": lookat,
        "eye": eye,
        "up": up,
        "field_of_view": fov_x,  # Use horizontal FOV, you could also use vertical FOV if preferred
        "intrinsic_matrix": intrinsic.intrinsic_matrix,
        "extrinsic_matrix": extrinsic
    }


def main(args):
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
    
    downsample = False

    if downsample:
        # Sub-sample the point cloud for better interactive experience
        for i, obj in enumerate(objects):
            pcd = obj['pcd']
            pcd = pcd.voxel_down_sample(0.01)
            # import pdb; pdb.set_trace()

            # points = np.asarray(pcd.points)
            # colors = np.asarray(pcd.colors)

            # point_cloud_df = pd.DataFrame(np.array(np.hstack((points, colors))), columns=list('xyzrgb'))

            # # drop uncolored points
            # colored_point_cloud_df = point_cloud_df.loc[point_cloud_df[list('rgb')].max(axis=1) > 0].reset_index()
            # colored_point_cloud_df['id'] = 0 # ID = 0 is not used for valid sparse map points
            # decimated_df = voxel_decimate(colored_point_cloud_df, 0.01)
            # merged_df = decimated_df

            # filtered_point_cloud_df = exclude_points(colored_point_cloud_df, sparse_point_cloud_df, radius=args.cell_size)
            # decimated_df = voxel_decimate(filtered_point_cloud_df, args.cell_size)

            # the dense points clouds have presumably more stable colors at corner points
            # rather use them than using the same approach as without dense data
            # sparse_colored_point_cloud_df = interpolate_missing_properties(colored_point_cloud_df, sparse_point_cloud_df[list('xyz')])
            # merged_df = pd.concat([sparse_colored_point_cloud_df, decimated_df])

            # if args.distance_quantile > 0:
            #     dist2 = (merged_df[list('xyz')]**2).sum(axis=1).values
            #     MARGIN = 1.5
            #     max_dist2 = np.quantile(dist2, args.distance_quantile) * MARGIN**2
            #     print(f'filtering out points further than {np.sqrt(max_dist2)}m')
            #     merged_df = merged_df.iloc[dist2 < max_dist2]

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(merged_df[list('xyz')].values)
            # pcd.colors = o3d.utility.Vector3dVector(merged_df[list('rgb')].values)

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

    def identify_desk(vis, query, T_max_bbox=None):
        max_prob_idx, max_pcd, max_bbox = color_by_clip_sim(vis, query=query)
        print("Most probable object is at index", max_prob_idx)
        print("max pcd:", max_pcd)
        print("max bbox:", max_bbox)
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

        geometries = []
        entities = [max_pcd]
        planes = []
        for obox in oboxes:
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
            mesh.paint_uniform_color(obox.color)

            print("Color: ", obox.color)

            vis.add_geometry(mesh)
            planes.append(mesh)

        # Determine which plane is the table top using T_max_bbox. 
        # The table top should be the plane that is closest to the robot bottom
        if T_max_bbox is not None:
            robot_center = T_max_bbox[:3, 3]
            min_dist = float('inf')
            min_id = None
            for obox_id, obox in enumerate(oboxes):
                normal = obox.R @ np.array([0, 0, 1])
                plane_center = obox.center

                # Normalize the normal vector
                n_norm = normal / np.linalg.norm(normal)

                # Calculate the distance t
                t = np.abs(np.dot(robot_center - plane_center, n_norm))

                if min_dist > t:
                    min_dist = t
                    min_id = obox_id
                    table_top = obox

            print("Table top is at index", min_id)
        else:
            table_top = oboxes[2]

        # get table_top normal
        # table_top_normal = table_top.R @ np.array([0, 0, 1])
        T_table_top = np.eye(4)
        T_table_top[:3, :3] = table_top.R
        T_table_top[:3, 3] = table_top.center

        # # side_normal = oboxes[2].R @ np.array([0, 0, 1])
        # T_side = np.eye(4)
        # T_side[:3, :3] = oboxes[1].R
        # T_side[:3, 3] = oboxes[1].center

        # # front_normal = oboxes[1].R @ np.array([0, 0, 1])
        # T_front = np.eye(4)
        # T_front[:3, :3] = oboxes[2].R
        # T_front[:3, 3] = oboxes[2].center

        # # add frame at table top where z points in the normal direction
        # frame_top = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])
        # frame_top.transform(T_table_top)

        # # add frame at side where z points in the normal direction
        # frame_side = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # frame_side.transform(T_side)

        # # add frame at front where z points in the normal direction
        # frame_front = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # frame_front.transform(T_front)

        return T_table_top, planes
    
    def identify_robot_bottom(vis, query="robot"):
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
        
        # get oriented bounding box
        max_bbox = max_pcd.get_oriented_bounding_box()
        
        print("Most probable object is at index", max_prob_idx)
        print("max pcd:", max_pcd)
        print("max bbox:", max_bbox)

        # Get pose of the bounding box
        T_max_bbox = np.eye(4)
        T_max_bbox[:3, :3] = max_bbox.R
        T_max_bbox[:3, 3] = max_bbox.center

        # Apply rotation in negative z-axis of 90 degrees
        T_rot = np.eye(4)
        T_rot[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        # Apply translation in positive y axis by the maximum extent of the bounding box
        T_y = np.eye(4)
        T_y[:3, 3] = np.array([0, np.max(max_bbox.extent) / 2.0, 0])
        T_max_bbox = T_max_bbox @ T_rot @ T_y

        return T_max_bbox, max_bbox, max_pcd
    
    def transform_robot_to_origin(T_table_top, T_max_bbox):
        # Place the table top frame at the center of the bounding box while staying on the table top
        T_01 = T_table_top.copy()
        T_02 = T_max_bbox.copy()

        T_12 = np.linalg.inv(T_01) @ T_02
        diff = T_12[:, 3]
        diff[2] = 0.0

        global_diff = T_01 @ diff

        T_table_top = np.eye(4)
        T_table_top[:3, :3] = T_01[:3, :3]
        T_table_top[:3, 3] = global_diff[:3]

        # Note this depends on which table top is detected and the the bounding box of the robot
        # TODO: Make this more robust
        # T_robot = T_table_top

        # # Rotate 90 degrees in negative z-axis
        # T_rot = np.eye(4)
        # T_rot[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        # T_robot = T_table_top @ T_rot

        # Rotate 180 degrees in positive y
        T_rot = np.eye(4)
        T_rot[:3, :3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        T_robot = T_table_top @ T_rot

        # Rotate 90 degrees in negative z
        T_rot = np.eye(4)
        T_rot[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        T_robot = T_robot @ T_rot

        # # Move the robot frame by 0.05 in the y-axis
        # T_trans = np.eye(4)
        # T_trans[:3, 3] = np.array([0, 0.05, 0])
        # T_robot = T_robot @ T_trans

        return np.linalg.inv(T_robot)

    franka = True
    revision = True

    if franka:
        # draw a frame at the origin for reference
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(frame)

        # fr3_path = "/home/lukas/Projects/concept-data/record3D/robot_small_preprocessed/fr3_franka.ply"
        # robot_pcd = visualize_fr3(vis, fr3_path)

        query = "robot"
        T_max_bbox, max_bbox, max_pcd = identify_robot_bottom(vis, query)

        # # Show frame of the bounding box for reference
        # bbox_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # bbox_frame.transform(T_max_bbox)

        # # remove the bounding boxes
        # for bbox in bboxes:
        #     vis.remove_geometry(bbox)

        # vis.add_geometry(bbox_frame)
        # vis.add_geometry(max_bbox)

        # max_pcd.paint_uniform_color([0.0, 0.0, 1.0])
        # vis.add_geometry(max_pcd)

        # vis.run()
        # exit()

        if revision:
            # query = "robot table"
            # query = "table"
            query = "desk"
        else:
            query = "box"
        T_table_top, planes = identify_desk(vis, query, T_max_bbox)
        # T_table_top, planes = identify_desk(vis, query)

        T_robot = transform_robot_to_origin(T_table_top, T_max_bbox)

        print("T_robot: ", T_robot)

        # # rotate the robot frame by negative 90 degrees in z-axis
        # T_rot = np.eye(4)
        # T_rot[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        # T_robot = T_rot @ T_robot

        # # move the robot fram by 0.5 in the x-axis
        # T_trans = np.eye(4)
        # T_trans[:3, 3] = np.array([0.5, 0, 0])
        # T_robot = T_trans @ T_robot 

        # Move the robot frame to the center of the bounding box
        # Add geometry to the scene
        for geometry in pcds:
            geometry.transform(T_robot)

        # queries = []
        # # queries = ["laptop", "books"]
        # for query in queries:
        #     max_prob_idx, max_pcd, max_bbox = color_by_clip_sim(vis, query=query)

        #     # transform the pcd to the robot frame
        #     transformed_pcd = copy.deepcopy(max_pcd).transform(np.linalg.inv(T_robot))
        #     points = np.asarray(transformed_pcd.points)

        #     print("x min: ", np.min(points[:, 0]))
        #     print("x max: ", np.max(points[:, 0]))
        #     print("y min: ", np.min(points[:, 1]))
        #     print("y max: ", np.max(points[:, 1]))
        #     print("z min: ", np.min(points[:, 2]))
        #     print("z max: ", np.max(points[:, 2]))

        #     # save point cloud to npy file
        #     np.save("{}_pcd_robot_frame.npy".format(query), points)

        #     # draw geometries
        #     o3d.visualization.draw_geometries([transformed_pcd])

        #     # save point cloud to ply file without colors
        #     new_pcd = o3d.geometry.PointCloud()
        #     new_pcd.points = o3d.utility.Vector3dVector(points)
        #     o3d.io.write_point_cloud("{}_pcd_robot_frame.ply".format(query), new_pcd)

        # Color the object based on RGB
        color_by_rgb(vis)

        # Save the transformed point clouds of the scene to a ply file
        filename = os.path.realpath(result_path).split(".")[0].split(".")[0]
        for i, pcd in enumerate(pcds):
            o3d.io.write_point_cloud("{}_{}.ply".format(filename, i), pcd)

        # Transform planes
        for plane in planes:
            plane.transform(T_robot)
            # vis.add_geometry(plane)

        # remove the frames
        frames = [frame]
        # for f in frames:
        #     vis.remove_geometry(f)

        # remove the planes
        for p in planes:
            vis.remove_geometry(p)

        # best_grasp_pose = np.load("/home/lukas/Projects/anygrasp/grasp_detection/example_data/best_grasp_pose.npy")
        print(result_path)
        scene_id = result_path.split("/")[-4:-3][0]
        data_dir_base = "/".join(result_path.split("/")[:-4])
        data_dir = "{}/{}/".format(data_dir_base, scene_id)
        print(data_dir)
        min_z = 0.1
        
        segmented_poses_filenames = [f for f in os.listdir(data_dir) if f.startswith("segmented_poses")]
        for segmented_poses_filename in segmented_poses_filenames:
            image_index = segmented_poses_filename.split("_")[2]
            print("image_index: ", image_index)
            text_prompt = "_".join(segmented_poses_filename.split("_")[3:])
            text_prompt = text_prompt.split(".")[0]
            print("text_prompt: ", text_prompt)
            segmented_grasp_poses = np.load(os.path.join(data_dir, segmented_poses_filename))
            camera_pose = np.load(os.path.join(data_dir, "map2c_{}.npy".format(image_index)))

            # Add camera frame
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            camera_frame.transform(T_robot @ np.linalg.inv(camera_pose))
            vis.add_geometry(camera_frame)

            # Transformation by 90 degrees in the positive y axis
            T_rot_90_y = np.eye(4)
            T_rot_90_y[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

            num_grasp_poses = segmented_grasp_poses.shape[0]
            T_grasps = np.zeros((num_grasp_poses, 4, 4))

            for i in range(num_grasp_poses):
                T_grasp = T_robot @ np.linalg.inv(camera_pose) @ segmented_grasp_poses[i] @ T_rot_90_y
                T_grasps[i] = T_grasp

            # Find the grasp where the z axis of the world frame is best aligned with the z axis of the grasp frame
            neg_z_axis = np.array([0, 0, -1])
            best_grasp_idx = None
            best_dot = -1
            for i in range(num_grasp_poses):
                grasp_z = T_grasps[i][:3, 2]
                dot = np.dot(neg_z_axis, grasp_z)
                if dot > best_dot:
                    best_dot = dot
                    best_grasp_idx = i

            T_grasp = T_grasps[best_grasp_idx]

            # Make sure the grasp has sufficient height over the table top
            print(T_grasp[2, 3])
            T_grasp[2, 3] = max(T_grasp[2, 3], min_z)
            print("T_grasp {}:".format(text_prompt), T_grasp)
            # save to npy file
            grasp_dir = os.path.join("/".join(result_path.split("/")[:-3]), "T_grasp_{}.npy".format(text_prompt))
            np.save(grasp_dir, T_grasp)

            # for T_grasp in T_grasps:
            for T_grasp in [T_grasps[best_grasp_idx]]:
                # draw a frame at the grasp pose
                frame_grasp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # Move the grasp by a 0.6 in the y-axis
                # T_trans = np.eye(4)
                # T_grasp[:3, 3] += np.array([0, 0.8, 0])
                frame_grasp.transform(T_grasp)
                vis.add_geometry(frame_grasp)

            # Create a pose for placing the object
            T_place = T_grasp.copy()

            if "paper_cup" in text_prompt:
                # T_place[:3, 3] += np.array([0.6, 0.75, 0])
                T_place[:3, 3] += np.array([0.5, 0.75, 0])
            elif "sponge" in text_prompt:
                T_place[:3, 3] += np.array([-0.25, -0.6, 0])
                T_place[2, 3] = min_z

            # draw a frame at the place pose
            frame_place = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            frame_place.transform(T_place)
            vis.add_geometry(frame_place)

            # Save the place pose
            place_dir = os.path.join("/".join(result_path.split("/")[:-3]), "T_place_{}.npy".format(text_prompt))
            np.save(place_dir, T_place)

        # # remove the CAD model robot
        # vis.remove_geometry(robot_pcd)

        # remove the bounding boxes
        for bbox in bboxes:
            vis.remove_geometry(bbox)

        # color_by_clip_sim(vis, query="laptop", highlight_max=False)

        # Color the object based on RGB
        color_by_rgb(vis)

        # Save the transformed point clouds of the scene to a ply file
        print(os.path.realpath(result_path))
        filename = os.path.realpath(result_path).split(".")[0].split(".")[0]
        for i, pcd in enumerate(pcds):
            o3d.io.write_point_cloud("{}_{}.ply".format(filename, i), pcd)

        # Load the saved camera parameters
        camera_params = load_camera_params("camera_params.json")
        converted_params = convert_camera_params(camera_params)
        o3d.visualization.draw(pcds, show_skybox=False, 
                                        bg_color=[1, 1, 1, 1], lookat=converted_params["lookat"],
                                        eye=converted_params["eye"], up=-converted_params["up"],
                                        field_of_view=converted_params["field_of_view"],
                                        intrinsic_matrix=converted_params["intrinsic_matrix"],
                                        extrinsic_matrix=converted_params["extrinsic_matrix"])
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
    parser = get_parser()
    args = parser.parse_args()
    main(args)

'''

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_ram_class_ram_stride50_no_bg__ram_class_ram_stride50_no_bg_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_ram__yolo_class_ram_stride50_no_bg4__ram_yolo_class_ram_stride50_no_bg_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz


python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_ram_class_ram_stride50_no_bg__TEST_ram_class_ram_stride50_no_bg_overlap_maskconf0.25_simsum1.2_dbscan.1.pkl.gz


python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_scannet200_class_ram_stride50_yes_bg2_mapping_scannet200_class_ram_stride50_yes_bg2.pkl.gz

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_38/full_pcd_s_mapping_yes_bg_38.pkl.gz

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_39/full_pcd_s_mapping_yes_bg_39_post.pkl.gz


python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_40/full_pcd_s_mapping_yes_bg_40_post.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_41/full_pcd_s_mapping_yes_bg_41_post.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_42/full_pcd_s_mapping_yes_bg_42_post.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_43/full_pcd_s_mapping_yes_bg_43_post.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/office0/exps/s_mapping_yes_bg_multirun_45/full_pcd_s_mapping_yes_bg_multirun_45.pkl.gz


python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/s_mapping_yes_bg_multirun_45/full_pcd_s_mapping_yes_bg_multirun_45.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/office0/exps/s_mapping_yes_bg_multirun_45/full_pcd_s_mapping_yes_bg_multirun_45.pkl.gz



python concept-graphs/conceptgraph/scripts/streamlined_detections.py

kernprof -l concept-graphs/conceptgraph/slam/streamlined_mapping.py
'''

