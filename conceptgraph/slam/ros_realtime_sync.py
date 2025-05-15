'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import logging
import os
from pathlib import Path
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
import time

# Third-party imports
import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import hydra
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv

# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, 
    orr_log_annotated_image, 
    orr_log_camera, 
    orr_log_depth_image, 
    orr_log_edges, 
    orr_log_objs_pcd_and_bbox, 
    orr_log_rgb_image, 
    orr_log_vlm_image
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import get_obj_rel_from_image_gpt4v, get_openai_client
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, 
    find_existing_image_path, 
    get_det_out_path, 
    get_exp_out_path, 
    get_vlm_annotated_image_path, 
    handle_rerun_saving, 
    load_saved_detections, 
    load_saved_hydra_json_config, 
    make_vlm_edges, 
    measure_time, 
    save_detection_results, 
    save_hydra_config, 
    save_objects_for_frame, 
    save_pointcloud, 
    should_exit_early, 
    vis_render_image
)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    OnlineObjectRenderer, 
    save_video_from_frames, 
    vis_result_fast_on_depth, 
    vis_result_for_vlm, 
    vis_result_fast, 
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, ProbabilisticMapObjectList, POCDObjectTypes
from conceptgraph.slam.utils_no_sampling import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections
from conceptgraph.dataset.conceptgraphs_datautils import scale_intrinsics
from conceptgraph.occupancygrid.utils import add_objects_to_occupancy_grid, dilate_map, show_occupancy_grid
from conceptgraph.llms.llama_client import LlamaClient, validate_output_pocd, validate_output_similarity
from conceptgraph.llms.prompts import POCD_SYSTEM_PROMPT, HEATMAP_SYSTEM_PROMPT, OBJECT_SIMILARITY_SYSTEM_PROMPT
from conceptgraph.utils.query_service_provider import QueryServiceProvider
from conceptgraph.utils.heatmap_publisher import HeatmapProvider



from ultralytics.engine.model import Model
import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import Image as ROSImage, CameraInfo, PointCloud2, Joy
from geometry_msgs.msg import PoseStamped, Point
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import ros2_numpy.point_cloud2 as point_cloud2
from message_filters import Subscriber as MF_Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import OccupancyGrid
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


DEBUG = True
# Disable torch gradient computation
torch.set_grad_enabled(False)


class Subscriber(Node):
    def __init__(self, cfg):
        super().__init__('subscriber')
        self.cfg = cfg

        self.use_SAI = True

        self.sub_joy = self.create_subscription(Joy, 'gamepad_joy', self._joy_callback, 1)

        topic_camera_info = 'spectacular_ai/camera_info' if self.use_SAI else 'camera/color/camera_info'
        self.sub_info = MF_Subscriber(self, CameraInfo, topic_camera_info)
        topic_color = 'spectacular_ai/color_image' if self.use_SAI else 'camera/color/image_raw'
        self.sub_color = MF_Subscriber(self, ROSImage, topic_color)

        if cfg.use_pc_for_depth:
            topic_depth = 'spectacular_ai/point_cloud/local' if self.use_SAI else 'camera/depth/points'
            self.sub_depth = MF_Subscriber(self, PointCloud2, topic_depth)
        else:
            topic_depth = 'spectacular_ai/depth_image' if self.use_SAI else 'camera/depth/image_raw'
            self.sub_depth = MF_Subscriber(self, ROSImage, topic_depth)
        
        self.MAX_MESSAGE_DELAY = 1/30
        self.callback_synchronizer = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 1, self.MAX_MESSAGE_DELAY)
        self.callback_synchronizer.registerCallback(self.callback_sync)
        # self.sub_pose = MF_Subscriber(self, PoseStamped, 'spectacular_ai/pose_image_synced')
        # self.callback_synchronizer = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth, self.sub_pose], 1, MAX_MESSAGE_DELAY)
        # self.callback_synchronizer.registerCallback(self.callback_sync_sai)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub_map = self.create_subscription(OccupancyGrid, 'map', self._map_callback, 1)
        self.pub_map = self.create_publisher(OccupancyGrid, 'map/conceptgraph', 1)
        self.pub_pc = self.create_publisher(PointCloud2, 'map/objects', 1)

        self.time_info_color_depth_ = None
        self.ready_info_color_depth_ = False
        self.map_with_button = cfg.map_with_button
        self.should_map = True

        self.info = None
        self.color = None
        self.depth = None
        self.pose = None
        self.changes = np.array([])
        self.map = None
        self.map_info = None

        self.objects = None

    def is_ready(self):
        if self.ready_info_color_depth_:
            self.pose = self._get_pose(time=self.time_info_color_depth_)
            return self.pose is not None
        else:
            return False
    
    def reset_ready(self):
        self.ready_info_color_depth_ = False
        self.ready_pose_ = False

    def callback_sync(self, info_msg, color_msg, depth_msg):
        self.ready_info_color_depth_ = False
        self.info, self.color, self.depth = self._process_inputs(info_msg, color_msg, depth_msg)
        should_map = (not self.map_with_button) or self.should_map # PS4 home button
        self.time_info_color_depth_ = color_msg.header.stamp
        self.ready_info_color_depth_ = should_map

    def callback_sync_sai(self, info_msg, color_msg, depth_msg, pose_msg):
        self.ready_info_color_depth_ = False
        self.info, self.color, self.depth = self._process_inputs(info_msg, color_msg, depth_msg)
        self.pose = self._process_pose_sai(pose_msg)
        self.ready_info_color_depth_ = True

    def publish_object_point_clouds(self):
        if self.objects is None:
            return
        # Create a PointCloud2 message
        self.objects.color_by_instance()
        pc_msg = self.objects.get_point_cloud_msg(frame_id="map")
        self.pub_pc.publish(pc_msg)

    def _joy_callback(self, msg):
        self.should_map = msg.axes[-1] == 1

    def _map_callback(self, msg):
        self.map_info = dict()
        self.map_info["resolution"] = msg.info.resolution
        self.map_info["origin"] = (msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.position.z)
        self.map_info["width"] = msg.info.width
        self.map_info["height"] = msg.info.height
        self.map = np.array(msg.data).reshape(msg.info.height, msg.info.width)

        if self.objects is None:
            return
        grid = add_objects_to_occupancy_grid(self.map, self.map_info, self.objects, max_height=3)
        new_msg = OccupancyGrid()
        new_msg.header = msg.header
        new_msg.info = msg.info
        new_msg.data = grid.flatten().tolist()
        self.pub_map.publish(new_msg)        

    def _process_inputs(self, info_msg, color_msg, depth_msg):
        # Process all inputs
        intrinsics = self._process_intrinsics(info_msg)
        color = self._process_color(color_msg)
        if self.cfg.use_pc_for_depth:
            depth = self._process_depth_from_pc(depth_msg, info_msg) # Info Message needed to project using K-matrix
        else:
            depth = self._process_depth(depth_msg)
        return intrinsics, color, depth
    
    def _process_color(self, color_msg):
        # Get data
        color = np.array(color_msg.data).astype(np.uint8).reshape(self.cfg["camera_params"]["image_height"], self.cfg["camera_params"]["image_width"], 3)
        # Resize
        color = cv2.resize(
            color,
            (self.cfg.desired_width, self.cfg.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        # Rotate if necessary
        if self.cfg.rotate_image:
            color = np.rot90(color, -1)
        # Convert to RGB from BGR
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        # Convert to torch tensor
        color = torch.from_numpy(color)
        color = color.to(self.cfg.device).type(torch.float)
        return color

    def _process_depth(self, depth_msg):
        # Get data
        if self.cfg.use_pc_for_depth:
            depth = depth_msg
        else:
            depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(self.cfg["camera_params"]["image_height"], self.cfg["camera_params"]["image_width"])
        invalid_indices = (depth < self.cfg.min_depth * 1000) | (depth > self.cfg.max_depth * 1000)
        depth[invalid_indices] = 0
        # Resize
        depth = cv2.resize(
            depth.astype(float),
            (self.cfg.desired_width, self.cfg.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        # Rotate if necessary
        if self.cfg.rotate_image:
            depth = np.rot90(depth, -1)
        # Clip depth
        # Convert depth to metres
        depth = depth / self.cfg["camera_params"]["png_depth_scale"]
        # Convert to torch tensor
        depth = torch.from_numpy(depth)
        depth = depth.to(self.cfg.device).type(torch.float)
        return depth

    def _process_depth_from_pc(self, pc_msg, info_msg):
        pc = point_cloud2.pointcloud2_to_array(pc_msg)
        # Format into numpy array
        points_camera = np.zeros((len(pc), 3))
        for i, point in enumerate(pc):
            points_camera[i] = point[0], point[1], point[2]
        # Project points onto the 2D image plane using intrinsics
        intrinsic_matrix = np.array(info_msg.k).reshape(3, 3)
        uv = (intrinsic_matrix @ points_camera.T).T
        uv[:, 0] /= uv[:, 2]
        uv[:, 1] /= uv[:, 2]
        # Keep only valid points that are in front of the camera, inside the image frame, and within the depth range
        valid_indices = (uv[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < self.cfg["camera_params"]["image_width"]) & \
                        (uv[:, 1] >= 0) & (uv[:, 1] < self.cfg["camera_params"]["image_height"]) & \
                        (uv[:, 2] >= self.cfg.min_depth) & (uv[:, 2] <= self.cfg.max_depth)
        # Create depth image
        depth_image = np.zeros((self.cfg["camera_params"]["image_height"], self.cfg["camera_params"]["image_width"]), dtype=np.uint16)
        uv_valid = uv[valid_indices]
        depth_valid = uv_valid[:, 2] * self.cfg["camera_params"]["png_depth_scale"]  # Scale depth to millimeters
        x_valid = uv_valid[:, 0].astype(np.int32)
        y_valid = uv_valid[:, 1].astype(np.int32)

        depth_image[y_valid, x_valid] = depth_valid.astype(np.uint16)
        # Process depth image
        depth = self._process_depth(depth_image)
        return depth
    
    def _process_pose(self, transform_msg):
        # Convert position + quaternion to pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat([
            transform_msg.transform.rotation.x,
            transform_msg.transform.rotation.y,
            transform_msg.transform.rotation.z,
            transform_msg.transform.rotation.w
        ]).as_matrix()
        pose[:3, 3] = np.array([
            transform_msg.transform.translation.x,
            transform_msg.transform.translation.y,
            transform_msg.transform.translation.z
        ])
        # Rotate if necessary
        if self.cfg.rotate_image:
            image_rotation = np.eye(4)
            image_rotation[:3, :3] = R.from_euler('z', -90, degrees=True).as_matrix()
            pose = pose @ image_rotation
        # Convert to torch tensor
        pose = torch.from_numpy(pose)
        pose = pose.to(self.cfg.device).type(torch.float)
        return pose
    
    def _process_pose_sai(self, pose_msg):
        # Convert position + quaternion to pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat([
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        ]).as_matrix()
        pose[:3, 3] = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z
        ])
        # Rotate if necessary
        if self.cfg.rotate_image:
            image_rotation = np.eye(4)
            image_rotation[:3, :3] = R.from_euler('z', -90, degrees=True).as_matrix()
            pose = pose @ image_rotation
        # Convert to torch tensor
        pose = torch.from_numpy(pose)
        pose = pose.to(self.cfg.device).type(torch.float)
        return pose 
    
    def _process_intrinsics(self, info_msg):
        # Get camera intrinsics and convert to torch tensor
        K = np.array(info_msg.k).reshape(3, 3)
        K = torch.from_numpy(K)
        # Scale intrinsics
        height_downsample_ratio = float(self.cfg.desired_height) / self.cfg["camera_params"]["image_height"]
        width_downsample_ratio = float(self.cfg.desired_width) / self.cfg["camera_params"]["image_width"]
        K = scale_intrinsics(K, height_downsample_ratio, width_downsample_ratio)
        # Rotate if necessary
        if self.cfg.rotate_image:
            K_tmp = K.clone()
            K[0, 2], K[1, 2] = K_tmp[1, 2], K_tmp[0, 2] # switch cx, cy
            K[0, 0], K[1, 1] = K_tmp[1, 1], K_tmp[0, 0] # switch fx, fy
        
        # Convert to torch tensor (not sure why we do this but its in the original dataset loader)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K
        intrinsics = intrinsics.to(self.cfg.device).type(torch.float)
        return intrinsics
    
    def _get_pose(self, time=None):
        if time is None:
            time = self.get_clock().now()
        if type(time) != rclpy.time.Time:
            time = rclpy.time.Time.from_msg(time)

        try:
            transform_msg = self.tf_buffer.lookup_transform("map", "camera_color_optical_frame", time)
            # Check that transform is recent
            transform_time = rclpy.time.Time.from_msg(transform_msg.header.stamp)
            diff = abs(time.nanoseconds - transform_time.nanoseconds) / 1e9
            if diff > self.MAX_MESSAGE_DELAY:
                print(colored(f"Transform is too old! Difference is {diff}. Maximum allowed delay is {self.MAX_MESSAGE_DELAY}.", 'red'), flush=True)
                return None
            
            return self._process_pose(transform_msg)
        except Exception as e:
            return None
        

def detect_objects(image_rgb: np.ndarray, frame_idx: int, detection_model: Model, sam_predictor: Model, cfg: DictConfig, obj_classes: ObjectClasses, clip_model, clip_preprocess, clip_tokenizer, tracker):
    results = None

    # opencv can't read Path objects...
    blur_score = cv2.Laplacian(image_rgb, cv2.CV_64F).var()
    if blur_score < cfg.blur_threshold:
        print(colored(f"Frame {frame_idx} is too blurry, skipping...\n" * 10, 'red'), flush=True)
        return None, None
    
    # Convert the numpy array to a PIL Image
    image_pil = Image.fromarray(image_rgb)

    # Do initial object detection
    results = detection_model.predict(image_pil, conf=0.1, verbose=False)
    confidences = results[0].boxes.conf.cpu().numpy()
    detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
    xyxy_tensor = results[0].boxes.xyxy
    xyxy_np = xyxy_tensor.cpu().numpy()

    # if there are detections,
    # Get Masks Using SAM or MobileSAM
    # UltraLytics SAM
    if xyxy_tensor.numel() != 0:
        sam_out = sam_predictor.predict(image_pil, bboxes=xyxy_tensor, verbose=False)
        masks_tensor = sam_out[0].masks.data

        masks_np = masks_tensor.cpu().numpy()
    else:
        masks_np = np.empty((0, *image_rgb.shape[:2]), dtype=np.float64)

    # Create a detections object that we will save later
    curr_det = sv.Detections(
        xyxy=xyxy_np,
        confidence=confidences,
        class_id=detection_class_ids,
        mask=masks_np,
    )
    if curr_det.xyxy.size == 0:
        print(f"No detections found for frame {frame_idx}", flush=True)
        return None, None

    image_crops, image_feats_gpu, text_feats = compute_clip_features_batched(
        image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

    # increment total object detections
    tracker.increment_total_detections(len(curr_det.xyxy))

    results = {
        # add new uuid for each detection 
        "xyxy": curr_det.xyxy,
        "confidence": curr_det.confidence,
        "class_id": curr_det.class_id,
        "mask": curr_det.mask,
        "classes": obj_classes.get_classes_arr(),
        "image_crops": image_crops,
        "image_feats": image_feats_gpu,
        "text_feats": text_feats,
        "detection_class_labels": detection_class_labels,
        # "labels": labels,
        # "edges": edges,
        "labels": [],
        "edges": [],
    }

    return results, curr_det

# A logger for this file
# @hydra.main(version_base=None, config_path="../hydra_configs/", config_name="ros_stretch")
# @profile
def main():
    # Initialize Hydra manually
    with hydra.initialize(config_path="../hydra_configs", version_base=None):
        cfg = hydra.compose(config_name="ros_stretch")
    print(OmegaConf.to_yaml(cfg), flush=True)

    logging.getLogger().setLevel(level=logging.INFO)
    tracker = MappingTracker()
    
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    orr.spawn()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs", 
            #    entity="concept-graphs",
                config=cfg_to_dict(cfg),
               )
    cfg = process_cfg(cfg)

    objects = ProbabilisticMapObjectList(device=cfg.device)
    objects_missing = ProbabilisticMapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # output folder for this mapping experiment
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)

    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False)

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )

    # if we need to do detections
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    
    prev_adjusted_pose = None

    if run_detections:
        print("\n".join(["Running detections..."] * 10), flush=True)
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        detection_model = measure_time(YOLO)(cfg.detection_model + '.pt')
        sam_predictor = SAM(cfg.sam_predictor_model + '.pt')
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())

        # LLM
        print("Setting up LLM...", flush=True)
        openai_client = get_openai_client()
        llamaClient = None
        if cfg.use_pocd_with_llm:
            llamaClient = LlamaClient(POCD_SYSTEM_PROMPT, max_tokens=20, output_validation_function=validate_output_pocd)
            pocd_type_cache = {}
        llamaClient_similarity = None
        if cfg.use_similarity_with_llm:
            llamaClient_similarity = LlamaClient(OBJECT_SIMILARITY_SYSTEM_PROMPT % obj_classes.get_classes_arr(), max_tokens=20, output_validation_function=validate_output_similarity)
        print("LLM setup complete.", flush=True)

        
    else:
        print("\n".join(["NOT Running detections..."] * 10), flush=True)

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, "saved_obj_all_frames")
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    if cfg.rotate_image:
        vis_camera_width = cfg.desired_height
        vis_camera_height = cfg.desired_width
    else:
        vis_camera_width = cfg.desired_width
        vis_camera_height = cfg.desired_height

    exit_early_flag = False
    counter = 0
    frame_idx = -1

    node = Subscriber(cfg=cfg)
    query_node = QueryServiceProvider(model = clip_model, tokenizer = clip_tokenizer, node=node)
    query_node.attach_objects(objects)
    heatmap_publisher = HeatmapProvider(
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        object_list=objects,
        missing_object_list=objects_missing,
        node=node,
        llm_client=llamaClient_similarity,
    )

    while rclpy.ok():
        # update reference of objects (filter method below creates a new object sometimes, TODO change this) 
        heatmap_publisher.object_list = objects
        heatmap_publisher.missing_object_list = objects_missing

        node.publish_object_point_clouds()
        while not node.is_ready():
            rclpy.spin_once(node, timeout_sec=0)

            # Manually initiate heatmap publishing
            if not hasattr(heatmap_publisher, "_last_update_time"):
                heatmap_publisher._last_update_time = time.time()

            current_time = time.time()
            if current_time - heatmap_publisher._last_update_time >= 1:
                heatmap_publisher.update_callback()
                heatmap_publisher._last_update_time = current_time
                print("Updating heatmap...", flush=True)
        node.reset_ready()

        local_time = time.time()
        frame_idx += 1
        tracker.curr_frame_idx = frame_idx
        counter += 1
        orr.set_time_sequence("frame", frame_idx)

        color_tensor, depth_tensor, intrinsics, pose_tensor = node.color, node.depth, node.info, node.pose

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        # cv2.imshow("depth", depth_array)
        # cv2.waitKey(0)
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Store current frame image
        color_path = Path(cfg.color_path) / f"{frame_idx:06}.png"
        if not color_path.parent.exists():
            color_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(color_path), image_rgb)

        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        
        # vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        # vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)
        
        if run_detections:
            raw_gobs, current_detections = detect_objects(image_rgb, frame_idx, detection_model, sam_predictor, cfg, obj_classes, clip_model, clip_preprocess, clip_tokenizer, tracker)
            if raw_gobs is None:
                continue

            # save the detections if needed
            if cfg.save_detections:

                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, _ = vis_result_fast(image_rgb, current_detections, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)

                annotated_depth_image, _ = vis_result_fast_on_depth(depth_image_rgb, current_detections, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, raw_gobs)
        else:
            # Support current and old saving formats
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                raise FileNotFoundError(f"No detections found for frame {frame_idx}at paths \n{det_exp_pkl_path / color_path.stem} or \n{det_exp_pkl_path / f'{int(color_path.stem):06}'}.")

        # get pose, this is the untrasformed pose.
        unt_pose = pose_tensor
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, vis_camera_width, vis_camera_height, frame_idx)

        orr_log_rgb_image(color_path)
        orr_log_annotated_image(color_path, det_exp_vis_path)
        orr_log_depth_image(depth_tensor.cpu())
        # orr_log_vlm_image(vis_save_path_for_vlm)
        # orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")

        # resize and filter the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        gobs = filter_gobs(resized_gobs, image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        # no detections in this frame
        if len(gobs['mask']) == 0:
            continue

        # this helps make sure things like pillows on couches are separate objects
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs['mask'],
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=-1,
            device=cfg.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                # obj["pcd"] = init_process_pcd(
                #     pcd=obj["pcd"],
                #     downsample_voxel_size=cfg["downsample_voxel_size"],
                #     dbscan_remove_noise=cfg["dbscan_remove_noise"],
                #     dbscan_eps=cfg["dbscan_eps"],
                #     dbscan_min_points=cfg["dbscan_min_points"],
                #     run_dbscan=False,
                # )
                obj["pcd"] = obj["pcd"].voxel_down_sample(cfg["downsample_voxel_size"])
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx, local_time
        )

        intrinsics_np = intrinsics.cpu().numpy()
        # Note: Here we are passing in height as width (and vice-versa) since the images got flipped
        expected_inds, expected_ids = objects.expectedToObserve(adjusted_pose, intrinsics_np, cfg['camera_params']['image_width'], cfg['camera_params']['image_height'], cfg.min_depth, cfg.max_depth, cfg['pocd_visibility_threshold'])

        for obj in objects:
            obj['confidence_history'] += [obj['pocd_confidence']]
        for obj in objects_missing:
            obj['confidence_history'] += [obj['pocd_confidence']]

        if len(detection_list) == 0: # no detections, skip
            if len(expected_inds) > 0:
                # Objects have dissapeared. Update POCD probabilities
                for obj in objects:
                    obj['time_of_disappearance'] = local_time
                change_list = [cfg.pocd_default_change] * len(expected_inds)
                change_std_list = [cfg.pocd_default_change_std] * len(expected_inds)
                objects.updateProbability(change_list=change_list, std_change_list=change_std_list, ids=expected_ids, cap=cfg.pocd_response)
                pruned_object_inds, _ = objects.pruneObjectsByProbability(cfg.pocd_removal_threshold)
                objects.removeObjectsByIndex(pruned_object_inds)
            continue

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log({
                    "total_objects_so_far": tracker.get_total_objects(),
                    "objects_this_frame": len(detection_list),
                })
            
            if cfg.save_objects_all_frames:
                save_objects_for_frame(
                    obj_all_frames_out_path,
                    frame_idx,
                    objects,
                    # cfg.obj_min_detections,
                    0,
                    adjusted_pose,
                    color_path
                )
            continue 

        ### compute similarities and then merge
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'], 
            detection_list=detection_list, 
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )

        visual_sim = compute_visual_similarities(detection_list, objects)

        # maps each detection to one object (list containing the index of the matched object)
        match_obj_indices = match_detections_to_objects(
            match_method=cfg['match_method'],
            phys_bias=cfg['phys_bias'],
            spatial_sim=spatial_sim,
            visual_sim=visual_sim,
            detection_threshold=cfg['sim_threshold'],
            spatial_threshold=cfg['physical_threshold'],
            semantic_threshold=cfg['semantic_threshold'],
            prioritize_semantic_similarity=cfg['prioritize_semantic_sim'],
        )

        ##### Perform POCD update
        if cfg.use_pocd:
            # # Use LLM to learn the object type: Dynamic (0), Semi-Static (1), or Static (2)
            obj_class_list = [obj["class_name"] for obj in detection_list]
            objects_types = [POCDObjectTypes.DYNAMIC.value] * len(obj_class_list)
            if cfg.use_pocd_with_llm:
                cached_types = pocd_type_cache.keys()
                for i, c in enumerate(obj_class_list):
                    if c in cached_types:
                        objects_types[i] = pocd_type_cache[c]
                        continue

                    prompt = f"{c}\n"
                    response, confidence = llamaClient.run_voting(prompt, num_votes=3)
                    response = response.split(".")[0]
                    print(colored(f"Object: {c}, Response: {response}, Confidence: {confidence}"), 'blue', flush=True)

                    object_type = POCDObjectTypes.DYNAMIC.value
                    if response == "semi-static":
                        object_type = POCDObjectTypes.STATIC.value
                    elif response == "static":
                        object_type = POCDObjectTypes.STATIC.value
                    objects_types[i] = object_type
                    pocd_type_cache[c] = object_type
            
            if expected_inds:

                # get change for each expected object
                change_list, std_change_list, obj_transformations = objects.getChangesForExpectedObjects(detection_list=detection_list, 
                                                                                                         expected_object_indices=expected_inds, 
                                                                                                         match_indices=match_obj_indices, 
                                                                                                         default_change=cfg.pocd_default_change, 
                                                                                                         default_change_std=cfg.pocd_default_change_std)
                
                # set expected objects with no found match as disappeared
                for index, change in zip(expected_inds, change_list):
                    if change == cfg.pocd_default_change:
                        # objects[index]['type'] = POCDObjectTypes.DISSAPEARED
                        objects[index]['time_of_disappearance'] = local_time

                # Update POCD probabilities of expected objects
                objects.updateProbability(change_list, std_change_list, expected_ids, cap=cfg.pocd_response)
                
                # Remove expected objects based on POCD
                to_remove = set()
                pruned_object_inds, pruned_object_ids = objects.pruneObjectsByProbability(cfg.pocd_removal_threshold)
                to_remove.update(pruned_object_inds)

                # Removed objects are now "missing"
                # objects_missing += [objects[i] for i in to_remove]

                # Translate expected objects based on POCD.
                # When an expected object has low enough pocd score, consider that it might moved farther
                # Compare each of these objects to all objects which appeared at a similar time as this object disappeared
                pruned_object_inds, pruned_object_ids = objects.pruneObjectsByProbability(cfg.pocd_transformation_threshold, cfg.pocd_removal_threshold)
                
                # Transform to a detection, if detected
                # objects.transformObjectsToDetection(expected_object_indices=expected_inds, transform_list=obj_transformations, inds=pruned_object_inds)
                # Transform to similar objects in library
                pruned_object_inds = [i for i in pruned_object_inds if i not in match_obj_indices]
                dissapeared_match_indices, transform_list = objects.matchDissapearedObjectsToRecentObjects(cfg.look_back_time, cfg.look_forward_time, pruned_object_inds)
                objects.mergeObjectsWithRecentObjects(pruned_object_inds, dissapeared_match_indices)
                to_remove.update(dissapeared_match_indices)

                # Remove objects
                to_remove = [i for i in list(to_remove) if i is not None]
                to_remove.sort(reverse=True)
                for ind in to_remove:
                    objects_missing.append(objects[ind]) # why are objects which were merged added here?
                    objects.pop(ind)
                    locations_in_list = []

                    # update match_obj_indices accordingly
                    for i, match_ind in enumerate(match_obj_indices):
                        if match_ind is None:
                            continue
                        elif match_ind == ind:
                            # a detection was matched to this object (which is now being removed)
                            locations_in_list.append(i)
                        elif match_ind > ind:
                            match_obj_indices[i] -= 1
                    
                    # delete the entries of this object in match_obj_indices and the corresponding detections
                    locations_in_list = locations_in_list[::-1]
                    for i in locations_in_list:
                        match_obj_indices.pop(i)
                        detection_list.pop(i)

                # Reject detections that are outliers
                ### TODO: Outlier detection doesn't work (no outliers detected). Fix it or get rid of it.
                # pruned_detection_inds = []
                # for i, ind in enumerate(match_obj_indices):
                #     if ind is None:
                #         continue
                #     if objects[ind]['inlier']:
                #         continue
                #     print(colored(f"Rejecting detection {detection_list[i]['class_name']} as an outlier\n" * 10, 'magenta'), flush=True)
                #     pruned_detection_inds.append(i)
                # pruned_detection_inds.sort(reverse=True)
                # for i in pruned_detection_inds:
                #     detection_list.pop(i)
                #     match_obj_indices.pop(i)

                # Add back in objects bsed on POCD
                removed_matches, transforms = objects.matchRemovedObjectsToRecentObjects(objects_missing, cfg.look_back_time, cfg.look_forward_time)
                objects.reinstateRemovedObjects(objects_missing, removed_matches)

                to_remove = set()
                for i, match in enumerate(removed_matches):
                    if match is None:
                        continue
                    to_remove.add(i)
                to_remove = sorted(list(to_remove), reverse=True)
                for i in to_remove:
                    objects_missing.pop(i)

        ##### End POCD Update

        # Now merge the detected objects into the existing objects based on the match indices
        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_obj_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device']
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )

        # map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges)

        is_final_frame = False #frame_idx == len(dataset) - 1 ... Still needed for other function signatures

        ### Perform post-processing periodically if told so
        objects.updateAge(time=local_time)
        objects.updateLostTime(time=local_time)

        # Denoising
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )

        # Filtering
        if processing_needed(
            cfg["filter_interval"],
            cfg["run_filter_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects,
                map_edges=map_edges
            )

        # Merging
        if processing_needed(
            cfg["merge_interval"],
            cfg["run_merge_final_frame"],
            frame_idx,
            is_final_frame,
        ) and len(objects) > 0:
            objects, map_edges = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=cfg["make_edges"],
                map_edges=map_edges
            )
        orr_log_objs_pcd_and_bbox(objects, obj_classes)
        orr_log_edges(objects, map_edges, obj_classes)

        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                # cfg.obj_min_detections,
                0,
                adjusted_pose,
                color_path
            )
        
        if cfg.pocd_plot:
            plt.figure(0)
            plt.clf()
            data = []
            for obj in objects:
                first_idx = obj['image_idx'][0]
                history = [None] * first_idx + obj['confidence_history']
                data.append([history, obj['class_name'], obj['id'], obj['curr_obj_num'], first_idx])
            legend = []
            for d in data:
                plt.plot(d[0])
                legend.append(d[1] + str(d[2]))
            plt.legend(legend)
            plt.ylim([0, 1])
            plt.xlim(left=0)
            plt.xlabel('Frame Index')
            plt.ylabel('POCD Confidence')
            plt.title('POCD Confidence Over Time')

        plt.pause(0.1)

        ### Downsample
        for obj in objects:
            reduced_pcd = obj["pcd"].voxel_down_sample(cfg["downsample_voxel_size"])
            obj['pcd'] = reduced_pcd
            obj["n_points"] = len(reduced_pcd.points)

        node.objects = objects

        if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
            # save the pointcloud
            save_pointcloud(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                cfg=cfg,
                objects=objects,
                obj_classes=obj_classes,
                latest_pcd_filepath=cfg.latest_pcd_filepath,
                create_symlink=True,
                edges=map_edges
            )

        owandb.log({
            "frame_idx": frame_idx,
            "counter": counter,
            "exit_early_flag": exit_early_flag,
            "is_final_frame": is_final_frame,
        })

        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log({
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
                })
    # LOOP OVER -----------------------------------------------------

    owandb.finish()
    node.destroy_node()

if __name__ == "__main__":
    rclpy.init()
    main()
    rclpy.shutdown()
