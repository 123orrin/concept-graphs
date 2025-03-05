'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import os
import copy
import uuid
from pathlib import Path
import pickle
import gzip
import pdb
from termcolor import colored

# Third-party imports
import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
import open3d as o3d
from open3d.io import read_pinhole_camera_parameters
from omegaconf import DictConfig
import hydra
from omegaconf import DictConfig
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
from conceptgraph.utils.llm import POCDLLM
from transformers import pipeline

import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import Image as ROSImage, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, Point
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import ros2_numpy.point_cloud2 as point_cloud2
from message_filters import Subscriber as MF_Subscriber, ApproximateTimeSynchronizer
from lsy_interfaces.srv import ConceptGraphQuery

import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from enum import Enum


DEBUG = True

class Subscriber(Node):
    def __init__(self, cfg):
        super().__init__('subscriber')
        self.cfg = cfg

        self.sub_info = MF_Subscriber(self, CameraInfo, 'camera/color/camera_info')
        self.sub_color = MF_Subscriber(self, ROSImage, 'camera/color/image_raw')
        # self.sub_info = MF_Subscriber(self, CameraInfo, 'spectacular_ai/camera_info')
        # self.sub_color = MF_Subscriber(self, ROSImage, 'spectacular_ai/color_image')

        if cfg.use_pc_for_depth:
            self.sub_depth = MF_Subscriber(self, PointCloud2, 'camera/depth/points')
            # self.sub_depth = MF_Subscriber(self, PointCloud2, 'spectacular_ai/point_cloud/local')
        else:
            self.sub_depth = MF_Subscriber(self, ROSImage, 'camera/depth/image_raw')
            # self.sub_depth = MF_Subscriber(self, ROSImage, 'spectacular_ai/depth_image')
        

        MAX_MESSAGE_DELAY = 1/15
        self.callback_synchronizer = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 1, MAX_MESSAGE_DELAY)
        self.callback_synchronizer.registerCallback(self.callback_sync)
        # self.sub_pose = MF_Subscriber(self, PoseStamped, 'spectacular_ai/pose_image_synced')
        # self.callback_synchronizer = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth, self.sub_pose], 1, MAX_MESSAGE_DELAY)
        # self.callback_synchronizer.registerCallback(self.callback_sync_sai)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ready_to_process = False
        self.info = None
        self.color = None
        self.depth = None
        self.pose = None

    def callback_sync(self, info_msg, color_msg, depth_msg):
        self.ready_to_process = False
        self.info, self.color, self.depth = self._process_inputs(info_msg, color_msg, depth_msg)
        self.pose = self._get_pose(time=color_msg.header.stamp)
        if self.pose is not None:
            self.ready_to_process = True

    def callback_sync_sai(self, info_msg, color_msg, depth_msg, pose_msg):
        self.ready_to_process = False
        self.info, self.color, self.depth = self._process_inputs(info_msg, color_msg, depth_msg)
        self.pose = self._process_pose_sai(pose_msg)
        self.ready_to_process = True

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
        if self.cfg.rotate:
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
        # # Subsample
        # depth = depth.flatten()
        # new_depth = np.zeros_like(depth)
        # sample_indices = range(0, np.size(depth), self.cfg.subsample_interval)
        # new_depth[sample_indices] = depth[sample_indices]
        # new_depth = new_depth.reshape(self.cfg["camera_params"]["image_height"], self.cfg["camera_params"]["image_width"])
        # depth = new_depth
        # # Remove floor
        # sp = np.shape(depth)
        # depth[int(sp[0] * (7/8)):,:] = 0
        # Resize
        depth = cv2.resize(
            depth.astype(float),
            (self.cfg.desired_width, self.cfg.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        # Rotate if necessary
        if self.cfg.rotate:
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
        if self.cfg.rotate:
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
        if self.cfg.rotate:
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
        if self.cfg.rotate:
            K[0, 2], K[1, 2] = K[1, 2], K[0, 2] # switch cx, cy
            K[0, 0], K[1, 1] = K[1, 1], K[0, 0] # switch fx, fy
        # Convert to torch tensor (not sure why we do this but its in the original dataset loader)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K
        intrinsics = intrinsics.to(self.cfg.device).type(torch.float)
        return intrinsics
    
    def _get_pose(self, time=rclpy.time.Time()):
        try:
            transform_msg = self.tf_buffer.lookup_transform("map", "camera_color_optical_frame", time)
            return self._process_pose(transform_msg)
        except Exception as e:
            print(f"Failed to get pose: {e}")
            return None
    
class QueryNode(Node):
    def __init__(self):
        super().__init__('query_node')
        self.query_service = self.create_service(ConceptGraphQuery, 'conceptgraph_query_service', self.query_callback)

        self.clip_model = None
        self.clip_tokenizer = None
        self.objects = None

    def query_callback(self, request, response):
        if not self.objects:
            response.object_center = Point()
            return
        text_query = request.query
        text_queries = [text_query]
        
        text_queries_tokenized = self.clip_tokenizer(text_queries).to("cuda")
        text_query_ft = self.clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        
        # similarities = objects.compute_similarities(text_query_ft)
        objects_clip_fts = self.objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)

        max_prob_object = self.objects[max_prob_idx]
        center = max_prob_object["bbox"].center
        print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
        print(f"location xyz: {center}")

        object_center = Point()
        object_center.x, object_center.y, object_center.z = center
        response.object_center = object_center
        return response

    def _attach_model(self, model):
        self.clip_model = model

    def _attach_tokenizer(self, tokenizer):
        self.clip_tokenizer = tokenizer
    
    def _attach_objects(self, objects):
        self.objects = objects
    

# Disable torch gradient computation
torch.set_grad_enabled(False)

# A logger for this file
@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="ros_stretch")
# @profile
def main(cfg : DictConfig):
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
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        detection_model = measure_time(YOLO)('yolov8l-world.pt')
        # detection_model = measure_time(YOLO)('yolov8l-worldv2.pt')
        # sam_predictor = SAM('sam_l.pt') 
        # sam_predictor = SAM('mobile_sam.pt') # UltraLytics SAM
        # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
        sam_predictor = SAM('sam2_b.pt')  # UltraLytics SAM 2 base
        # sam_predictor = SAM('sam2_t.pt')  # UltraLytics SAM 2 tiny
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())

        # LLM
        print("Setting up LLM...")
        openai_client = get_openai_client()
        pocd_llm = POCDLLM(cfg.llm_model_id, num_reprompt_tries=3)
        print("LLM setup complete.")

        
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, "saved_obj_all_frames")
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    if cfg.rotate:
        vis_camera_width = cfg.desired_height
        vis_camera_height = cfg.desired_width
    else:
        vis_camera_width = cfg.desired_width
        vis_camera_height = cfg.desired_height

    exit_early_flag = False
    counter = 0
    frame_idx = -1

    node = Subscriber(cfg=cfg)
    query_service_node = QueryNode()
    query_service_node._attach_model(clip_model)
    query_service_node._attach_tokenizer(clip_tokenizer)
    query_service_node._attach_objects(objects)
    while rclpy.ok():
        
        while not node.ready_to_process:
            rclpy.spin_once(node, timeout_sec=0)
            rclpy.spin_once(query_service_node, timeout_sec=0)
        node.ready_to_process = False

        frame_idx += 1
        tracker.curr_frame_idx = frame_idx
        counter+=1
        orr.set_time_sequence("frame", frame_idx)

        color_tensor, depth_tensor, intrinsics, pose_tensor = node.color, node.depth, node.info, node.pose
        #color_tensor2, depth_tensor2, intrinsics2, *_ = dataset[frame_idx]

        # Read info about current frame from dataset
        # color image
        color_path = Path(cfg.color_path) / f"{frame_idx:06}.png"
        # Check if path exists up to the file name
        if not color_path.parent.exists():
            # Create the directory if it doesn't exist
            color_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(color_path), color_tensor.cpu().numpy())
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        # cv2.imshow("depth", depth_array)
        # cv2.waitKey(0)
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")
        
        # vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        # vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)
        
        if run_detections:
            results = None
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path)) # This will in BGR color space
            blur_score = cv2.Laplacian(image, cv2.CV_64F).var()
            if blur_score < cfg.blur_threshold:
                print(colored(f"Frame {frame_idx} is too blurry, skipping...\n" * 10, 'red'))
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Do initial object detection
            results = detection_model.predict(color_path, conf=0.1, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            if xyxy_tensor.numel() != 0:
                sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()
            else:
                masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)

            # Create a detections object that we will save later
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            if curr_det.xyxy.size == 0:
                print(f"No detections found for frame {frame_idx}")
                continue
            
            # Make the edges
            # print("")
            # print("MAKING EDGES MAKING EDGES MAKING EDGES")
            # print("")
            # pdb.set_trace()
            
            # labels, edges, edge_image = make_vlm_edges(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, cfg.make_edges, openai_client)
            # print("")
            # print("MADE EDGES MADE EDGES MADE EDGES")
            # print("")
            # pdb.set_trace()

            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection 
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                # "labels": labels,
                # "edges": edges,
                "labels": [],
                "edges": [],
            }

            raw_gobs = results

            # save the detections if needed
            if cfg.save_detections:

                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)

                annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
        else:
            # Support current and old saving formats
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                # if no detections, throw an error
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

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
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
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx # TODO: ADD TIME HERE
        )

        intrinsics_np = intrinsics.cpu().numpy()
        # Note: Here we are passing in height as width (and vice-versa) since the images got flipped
        expected_inds, expected_ids = objects.expectedToObserve(adjusted_pose, intrinsics_np, cfg['camera_params']['image_width'], cfg['camera_params']['image_height'], cfg.min_depth, cfg.max_depth)
        for i in expected_inds:
            print(colored(f"Expected to see {objects[i]['class_name']}", 'green'))

        if len(detection_list) == 0: # no detections, skip
            if len(expected_inds) > 0:
                # Objects have dissapeared. Update POCD probabilities
                change_list = [cfg.pocd_default_change] * len(expected_inds)
                change_std_list = [cfg.pocd_default_change_std] * len(expected_inds)
                objects.updateProbability(change=change_list, std_change=change_std_list, ids=expected_ids, cap=cfg.pocd_response)
                objects.pruneObjectsByProbability(cfg.pocd_removal_threshold)                
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
            continue 

        ### compute similarities and then merge
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'], 
            detection_list=detection_list, 
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )

        visual_sim = compute_visual_similarities(detection_list, objects)

        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            spatial_sim=spatial_sim, 
            visual_sim=visual_sim
        )

        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim, 
            detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
        )

        ##### Perform POCD update
        if cfg.use_pocd:
            match_ids = [objects[i]['id'] if i is not None else None for i in match_indices]
            # # Use LLM to learn the object type: Dynamic (0), Semi-Static (1), or Static (2)
            obj_class_list = [obj["class_name"] for obj in detection_list]
            # # object_type, retries = pocd_llm.run_inference(obj_class_list, max_response_length=200)
            # # object_type, retries = pocd_llm.run_inference_single(obj_class_list, max_response_length=100)
            object_type = [0] * len(obj_class_list)
            # retries = "NO LLM"
            # print(colored(f"LLM input: {obj_class_list}", 'green'))
            # print(colored(f"LLM output: {object_type}", 'green'))
            # print(colored(f"LLM retries: {retries}", 'red'))
            
            if expected_inds:
                change_list = [cfg.pocd_default_change] * len(expected_inds)
                std_change_list = [cfg.pocd_default_change_std] * len(expected_inds)
                transform_list = [np.eye(4)] * len(expected_inds)
                for detection_idx, object_idx in enumerate(match_indices):
                    if object_idx is None:
                        # Object is new. No POCD update
                        continue
                    if object_idx not in expected_inds:
                        # Object is not expected. No POCD update
                        continue

                    index = expected_inds.index(object_idx)
                    objects[object_idx]['type'] = POCDObjectTypes(object_type[detection_idx])
                    # Object has been observed. Evaluate change magnitude with ICP
                    detection_pcd = detection_list[detection_idx]['pcd']
                    object_pcd = objects[object_idx]['pcd']
                    transform_init = np.eye(4)
                    threshold = 0.01
                    registration_results = o3d.pipelines.registration.registration_icp(
                        detection_pcd, object_pcd, threshold, transform_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
                    std_change_list[index] = cfg.pocd_default_change_std
                    change_list[index] = np.linalg.norm(registration_results.transformation[:3, 3])
                    # std_change_list[index] = registration_results.inlier_rmse
                    transform_list[index] = registration_results.transformation
                
                for i, index in enumerate(expected_inds):
                    if change_list[i] == cfg.pocd_default_change:
                        objects[index]['type'] = POCDObjectTypes.DISSAPEARED
                        print(colored(f"Object {objects[index]['class_name']} has dissapeared", 'cyan'))

                objects.updateProbability(change_list, std_change_list, expected_ids, cap=cfg.pocd_response)
                # is_valid_detection = objects.getValidDetections(expected_ids) # Valid detection if measurement is an inlier
                
                # Remove objects based on POCD
                pruned_object_inds, pruned_object_ids = objects.pruneObjectsByProbability(cfg.pocd_removal_threshold)
                pruned_object_inds.sort(reverse=True)
                for i in pruned_object_inds:
                    print(colored(f"Removing object {objects[i]['class_name']} with probability {objects[i]['pocd_confidence']}", 'red'))
                    objects_missing.append(i)
                    objects.pop(i)
                
            #     # Translate objects based on POCD
            #     pruned_object_inds, pruned_object_ids = objects.pruneObjectsByProbability(cfg.pocd_transformation_threshold)
            #     for i in pruned_object_inds:
            #         print(colored(f"Transforming object {objects[i]['class_name']} with probability {objects[i]['pocd_confidence']}", 'yellow'))
            #         if i not in expected_inds:
            #             continue
            #         ind = expected_inds.index(i)
            #         transform = transform_list[ind]
            #         objects[i]['pcd'].transform(transform)
                    
            #         oriented_bbox = objects[i]['bbox'].get_oriented_bounding_box()
            #         oriented_bbox.translate(transform[:3, 3])
            #         oriented_bbox.rotate(transform[:3, :3])
            #         axis_bbox = oriented_bbox.get_axis_aligned_bounding_box()
            #         objects[i]['bbox'] = axis_bbox

            #     # Add back in objects bsed on POCD
            #     # TODO: Add back in objects based on POCD

            ### Fix other variables affected by POCD Update
                # Reject detections that have large changes
                for obj_ind in pruned_object_inds:
                    if obj_ind not in match_indices:
                        continue
                    ind = match_indices.index(obj_ind)
                    match_indices.pop(ind)
                    detection_list.pop(ind)
                    
            #     # Make the detection a new object if the previous objects were removed
            #     num_objects = len(objects)
            #     for i, ind in enumerate(match_indices):
            #         if ind is not None and ind >= num_objects:
            #             match_indices[i] = None
        ##### End POCD Update


        # Now merge the detected objects into the existing objects based on the match indices
        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device']
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )
        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges)

        is_final_frame = False #frame_idx == len(dataset) - 1 ... Still needed for other function signatures

        ### Perform post-processing periodically if told so

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

        query_service_node._attach_objects(objects)

        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                cfg.obj_min_detections,
                adjusted_pose,
                color_path
            )

        ### Downsample
        for obj in objects:
            reduced_pcd = obj["pcd"].voxel_down_sample(cfg["downsample_voxel_size"])
            obj['pcd'] = reduced_pcd
            obj["n_points"] = len(reduced_pcd.points)

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
    
    handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix, exp_out_path)

    # Save the pointcloud
    if cfg.save_pcd:
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

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': obj_classes.get_class_color_dict_by_index(),
            }, f)

    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    owandb.finish()
    node.destroy_node()

if __name__ == "__main__":
    rclpy.init()
    main()
    rclpy.shutdown()
