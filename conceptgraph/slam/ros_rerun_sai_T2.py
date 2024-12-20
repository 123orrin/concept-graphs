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
import time

# Third-party imports
import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv


os.environ["WANDB_MODE"] = "disabled" #fuxiao disable W&B

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
#from conceptgraph.utils.vlm import get_obj_rel_from_image_gpt4v, get_openai_client#fuxiao delete openai
from conceptgraph.utils.vlm import get_obj_rel_from_image_gpt4v
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
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, Point
import ros2_numpy.point_cloud2 as point_cloud2
# from lsy_interfaces.srv import ConceptGraphQuery

import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber')

        self.sub_color = self.create_subscription(ROSImage, 'spectacular_ai/color_image', self.color_callback, 10)
        self.sub_depth = self.create_subscription(ROSImage, 'spectacular_ai/depth_image', self.depth_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, 'spectacular_ai/camera_info', self.info_callback, 10)
        self.sub_pose = self.create_subscription(PoseStamped, 'spectacular_ai/pose_image_synced', self.pose_callback, 10)
        self.sub_pc = self.create_subscription(PointCloud2, 'spectacular_ai/point_cloud/local', self.pc_callback, 10)

        self.color_msg = None
        self.depth_msg = None
        self.camera_msg = None
        self.pose_msg = None
        self.pc_msg = None

        self.received_color = False
        self.received_depth = False
        self.received_info = False
        self.received_pose = False
        self.received_pc = False

    def color_callback(self, msg):
        self.color_msg = msg
        self.received_color = True

    def depth_callback(self, msg):
        self.depth_msg = msg
        self.received_depth = True

    def info_callback(self, msg):
        self.camera_msg = msg
        self.received_info = True

    def pose_callback(self, msg):
        self.pose_msg = msg
        self.received_pose = True
    
    def pc_callback(self, msg):
        self.pc_msg = msg
        self.received_pc = True

    def process_inputs(self, cfg, rotate=True, use_pc_for_depth=False): #rotate is set default true
        # Process all inputs
        color = self._process_color(cfg, rotate)
        if use_pc_for_depth:
            depth = self._process_depth_from_pc(cfg, rotate)
        else:
            depth = self._process_depth(cfg, rotate)
        intrinsics = self._process_intrinsics(cfg, rotate)
        pose = self._process_pose(cfg, rotate)
        # Set flags to false for next iteration
        self.received_color = False
        self.received_depth = False
        self.received_info = False
        self.received_pose = False
        self.received_pc = False
        # Return processed inputs
        return color, depth, intrinsics, pose
    
    def _process_color(self, cfg, rotate):
        # Get data
        color = np.array(self.color_msg.data).astype(np.uint8).reshape(cfg.true_height, cfg.true_width, 3) # pass true_height and true_width from cfg file
        # Rotate if necessary
        if rotate:
            color = np.rot90(color, -1)
        # Resize
        color = cv2.resize(
            color,
            (cfg.desired_width, cfg.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        # Convert to RGB from BGR
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR) #from RGB to BGR
        # Convert to torch tensor
        color = torch.from_numpy(color)
        color = color.to(cfg.device).type(torch.float)
        return color

    def _process_depth(self, cfg, rotate, from_pc=False, depth_from_pc=None):
        # Get data
        if from_pc:
            depth = depth_from_pc
        else:
            depth = np.frombuffer(self.depth_msg.data, dtype=np.uint16).reshape(cfg.true_height, cfg.true_width)
        # Rotate if necessary
        if rotate:
            depth = np.rot90(depth, -1)
        # Clip depth
        invalid_indices = (depth < cfg.min_depth) & (depth > cfg.max_depth)
        depth[invalid_indices] = 0
        # Resize
        depth = cv2.resize(
            depth.astype(float),
            (cfg.desired_width, cfg.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        # Convert depth to metres
        depth = depth / cfg["camera_params"]["png_depth_scale"]
        # Convert to torch tensor
        depth = torch.from_numpy(depth)
        depth = depth.to(cfg.device).type(torch.float)
        return depth

    def _process_depth_from_pc(self, cfg, rotate):
        # Get data
        if not self.received_info:
            return
        pc = point_cloud2.pointcloud2_to_array(self.pc_msg)
        # Format into numpy array
        points_camera = np.zeros((len(pc), 3))
        for i, point in enumerate(pc):
            points_camera[i] = point[0], point[1], point[2]
        # Project points onto the 2D image plane using intrinsics
        intrinsic_matrix = np.array(self.camera_msg.k).reshape(3, 3)
        uv = (intrinsic_matrix @ points_camera.T).T
        uv[:, 0] /= uv[:, 2]
        uv[:, 1] /= uv[:, 2]
        # Keep only valid points that are in front of the camera, inside the image frame, and within the depth range
        valid_indices = (uv[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cfg.true_width) & \
                        (uv[:, 1] >= 0) & (uv[:, 1] < cfg.true_height) & \
                        (uv[:, 2] >= cfg.min_depth) & (uv[:, 2] <= cfg.max_depth)
        # Create depth image
        depth_image = np.zeros((cfg.true_height, cfg.true_width), dtype=np.uint16)
        uv_valid = uv[valid_indices]
        depth_valid = uv_valid[:, 2] * cfg["camera_params"]["png_depth_scale"]  # Scale depth to millimeters
        x_valid = uv_valid[:, 0].astype(np.int32)
        y_valid = uv_valid[:, 1].astype(np.int32)

        depth_image[y_valid, x_valid] = depth_valid.astype(np.uint16)
        # Process depth image
        depth = self._process_depth(cfg, rotate, from_pc=True, depth_from_pc=depth_image)
        return depth
    
    def _process_pose(self, cfg, rotate):
        # Convert position + quaternion to pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat([
            self.pose_msg.pose.orientation.x,
            self.pose_msg.pose.orientation.y,
            self.pose_msg.pose.orientation.z,
            self.pose_msg.pose.orientation.w
        ]).as_matrix()
        pose[:3, 3] = np.array([
            self.pose_msg.pose.position.x,
            self.pose_msg.pose.position.y,
            self.pose_msg.pose.position.z
        ])
        # Rotate if necessary
        if rotate:
            image_rotation = np.eye(4)
            image_rotation[:3, :3] = R.from_euler('z', -90, degrees=True).as_matrix()
            pose = pose @ image_rotation
        # Convert to torch tensor
        pose = torch.from_numpy(pose)
        pose = pose.to(cfg.device).type(torch.float)
        return pose
    
    def _process_intrinsics(self, cfg, rotate):
        # Get camera intrinsics and convert to torch tensor
        K = np.array(self.camera_msg.k).reshape(3, 3)
        # Rotate if necessary
        if rotate:
            K[0, 2], K[1, 2] = K[1, 2], K[0, 2] # switch cx, cy
            K[0, 0], K[1, 1] = K[1, 1], K[0, 0] # switch fx, fy
        K = torch.from_numpy(K)
        # Scale intrinsics
        height_downsample_ratio = float(cfg.desired_height) / cfg.image_height
        width_downsample_ratio = float(cfg.desired_width) / cfg.image_width
        K = scale_intrinsics(K, height_downsample_ratio, width_downsample_ratio)
        # Convert to torch tensor (not sure why we do this but its in the original dataset loader)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K
        intrinsics = intrinsics.to(cfg.device).type(torch.float)
        return intrinsics
    
# class QueryNode(Node):
#     def __init__(self):
#         super().__init__('query_node')
#         self.query_service = self.create_service(ConceptGraphQuery, 'conceptgraph_query_service', self.query_callback)

#         self.clip_model = None
#         self.clip_tokenizer = None
#         self.objects = None

#     def query_callback(self, request, response):
#         if not self.objects:
#             response.object_center = Point()
#             return
#         text_query = request.query
#         text_queries = [text_query]
        
#         text_queries_tokenized = self.clip_tokenizer(text_queries).to("cuda")
#         text_query_ft = self.clip_model.encode_text(text_queries_tokenized)
#         text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
#         text_query_ft = text_query_ft.squeeze()
        
#         # similarities = objects.compute_similarities(text_query_ft)
#         objects_clip_fts = self.objects.get_stacked_values_torch("clip_ft")
#         objects_clip_fts = objects_clip_fts.to("cuda")
#         similarities = F.cosine_similarity(
#             text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
#         )
#         max_value = similarities.max()
#         min_value = similarities.min()
#         probs = F.softmax(similarities, dim=0)
#         max_prob_idx = torch.argmax(probs)

#         max_prob_object = self.objects[max_prob_idx]
#         center = max_prob_object["bbox"].center
#         print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
#         print(f"location xyz: {center}")

#         object_center = Point()
#         object_center.x, object_center.y, object_center.z = center
#         response.object_center = object_center
#         return response

#     def _attach_model(self, model):
#         self.clip_model = model

#     def _attach_tokenizer(self, tokenizer):
#         self.clip_tokenizer = tokenizer
    
#     def _attach_objects(self, objects):
#         self.objects = objects
    

# Disable torch gradient computation
torch.set_grad_enabled(False)

# A logger for this file
@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="ros_tableware")
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

    if cfg.rotate: #"rotate" seems to be defined in ros.yaml
        cfg.image_width, cfg.image_height = cfg.image_height, cfg.image_width
        cfg.desired_width, cfg.desired_height = cfg.desired_height, cfg.desired_width

    # print(cfg.end)

    # Initialize the dataset
    # dataset = get_dataset(
    #     dataconfig=cfg.dataset_config,
    #     start=cfg.start,
    #     end=cfg.end,
    #     stride=cfg.stride,
    #     basedir=cfg.dataset_root,
    #     sequence=cfg.scene_id,
    #     desired_height=cfg.image_height,
    #     desired_width=cfg.image_width,
    #     device="cpu",
    #     dtype=torch.float,
    # )
    # cam_K = dataset.get_cam_K()

    # # Load list of blurry images
    # skipped_frames_path = Path(cfg.dataset_root) / cfg.scene_id / "skipped_frames.txt"
    # with open(skipped_frames_path, "r") as f:
    #     skipped_frames = f.read().splitlines()
    # skipped_frames = [int(frame) for frame in skipped_frames]

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # For visualization
    if cfg.vis_render:
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param = view_param,
            base_objects = None, 
            gray_map = False,
        )
        frames = []
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
        sam_predictor = SAM('mobile_sam.pt') # UltraLytics SAM
        # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
        # sam_predictor = SAM('sam2.1_l.pt')  # UltraLytics SAM 2 large
        # sam_predictor = SAM('sam2.1_s.pt')  # UltraLytics SAM 2 small
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())

        #openai_client = get_openai_client()
        
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    frame_idx = -1

    start_time = time.time()  # Start the timer
    max_duration = 45  # Maximum duration in seconds

    node = Subscriber()
    # query_service_node = QueryNode()
    # query_service_node._attach_model(clip_model)
    # query_service_node._attach_tokenizer(clip_tokenizer)
    # query_service_node._attach_objects(objects)
    while rclpy.ok():
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        if elapsed_time >= max_duration:
            print(f"Stopping script after {max_duration} seconds.")
            break  # Exit the loop
        




        frame_idx += 1
        # if counter + 1 in skipped_frames:
        #     print(f"Skipping frame {frame_idx} as it is blurry")
        #     counter+=1
        #     continue
        tracker.curr_frame_idx = frame_idx
        counter+=1
        orr.set_time_sequence("frame", frame_idx)

        if cfg.use_pc_for_depth:
            while not (node.received_color and node.received_pc and node.received_info and node.received_pose):
                rclpy.spin_once(node, timeout_sec=0)
                # rclpy.spin_once(query_service_node, timeout_sec=0)
        else:
            while not (node.received_color and node.received_depth and node.received_info and node.received_pose):
                rclpy.spin_once(node, timeout_sec=0)
                # rclpy.spin_once(query_service_node, timeout_sec=0)

        #The process functions will be different for preparing the data for different pipeline
        color_tensor, depth_tensor, intrinsics, pose_tensor = node.process_inputs(cfg,
                                                                                  rotate=cfg.rotate,
                                                                                  use_pc_for_depth=cfg.use_pc_for_depth)
        
        #Manage paths for storing color img, depth img, camera poses, camera intrin

        save_paths = []
        
        if cfg.save_color:
            color_path = Path(cfg.color_path) / f"{frame_idx:06}.png"
            save_paths.append(color_path)
   
        if cfg.save_depth:
            depth_path = Path(cfg.depth_path) / f"{frame_idx:06}.png"
            save_paths.append(depth_path)

        if cfg.save_pose:
            pose_path = Path(cfg.pose_path) / f"{frame_idx:06}.npy"
            save_paths.append(pose_path)

        if cfg.save_intrinsics and frame_idx == 0:              
            intrinsics_path = Path(cfg.intrinsics_path) / f"{frame_idx:06}.npy"
            save_paths.append(intrinsics_path)

        # if cfg.save_intrinsics:              
        #     intrinsics_path = Path(cfg.intrinsics_path) / f"{frame_idx:06}.npy"
        #     save_paths.append(intrinsics_path)

        for save_path in save_paths:
            # Check if path exists up to the file name
            if not save_path.parent.exists():
                # Create the directory if it doesn't exist
                save_path.parent.mkdir(parents=True, exist_ok=True)

        # save data

        color_np = color_tensor.cpu().numpy() # (H, W, 3) np array
        if cfg.save_color:
            cv2.imwrite(str(color_path), color_np) # using opcv function to save the img into a path, use cpu, cv2.imwrite only support path in str format.
        # do some sanity checks
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]" # not sure if it is checking non-positive error?


        depth_tensor = depth_tensor[..., 0] # possibly (H,W,depth)
        depth_array = depth_tensor.cpu().numpy()
        if cfg.save_depth:
            cv2.imwrite(str(depth_path),(depth_array*1000).astype(np.unit16))
        # cv2.imshow("depth", depth_array)
        # cv2.waitKey(0)


        # get pose, this is the untrasformed pose.
        unt_pose = pose_tensor
        unt_pose = unt_pose.cpu().numpy()
        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        if cfg.save_pose:
            np.save(pose_path, adjusted_pose)
        
                # Save the camera intrinsics
        if cfg.save_intrinsics and frame_idx == 0:
            np.save(intrinsics_path, intrinsics.cpu().numpy())

        #         # Save the camera intrinsics
        # if cfg.save_intrinsics:
        #     np.save(intrinsics_path, intrinsics.cpu().numpy())


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

        # pdb.set_trace()


        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, cfg.desired_width, cfg.desired_height, frame_idx)
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
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )

        if len(detection_list) == 0: # no detections, skip
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

        # pdb.set_trace()

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

        is_final_frame = False #frame_idx == len(dataset) - 1
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

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
        ):
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

        # query_service_node._attach_objects(objects)

        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                cfg.obj_min_detections,
                adjusted_pose,
                color_path
            )
        
        if cfg.vis_render:
            # render a frame, if needed (not really used anymore since rerun)
            vis_render_image(
                objects,
                obj_classes,
                obj_renderer,
                image_original_pil,
                adjusted_pose,
                frames,
                frame_idx,
                color_path,
                cfg.obj_min_detections,
                cfg.class_agnostic,
                cfg.debug_render,
                is_final_frame,
                cfg.exp_out_path,
                cfg.exp_suffix,
            )

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
