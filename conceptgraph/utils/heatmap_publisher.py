import numpy as np
import torch
from torch.nn import functional as F
from rclpy.node import Node
from std_msgs.msg import String as StringMsg
from nav_msgs.msg import OccupancyGrid

from conceptgraph.slam.slam_classes import ProbabilisticMapObjectList
from conceptgraph.occupancygrid.utils import world_to_cell
from geometry_msgs.msg import Pose

class HeatmapProvider(Node):
    def __init__(self, clip_model, clip_tokenizer, object_list: ProbabilisticMapObjectList, missing_object_list: ProbabilisticMapObjectList):
        super().__init__("heatmap_publisher")

        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.object_list = object_list
        self.missing_object_list = missing_object_list

        self.device = "cuda"

        self.query_text = ""
        self.query_feature_dev = None
        self.occupancy_info = {
            "resolution": 0.05,  # meters per pixel
            "width": 3.0, # meters
            "height": 2.0,
            "origin": (-1, -1),
        }

        self.query_subscription = self.create_subscription(
            StringMsg, "heatmap_goal", self.query_callback, 1
        )
        self.heatmap_publisher = self.create_publisher(OccupancyGrid, "heatmap", 10)
        self.timer = self.create_timer(1, self.update_callback)

    def query_callback(self, msg):
        self.query_text = msg.data

        text_queries = [self.query_text]
        text_queries_tokenized = self.clip_tokenizer(text_queries).to("cuda")
        self.query_feature_dev = self.clip_model.encode_text(text_queries_tokenized)

        self.get_logger().info(f"Received query: {self.query_text}")

    def update_callback(self):
        if self.query_text == "":
            return

        # Create a heatmap from the object list
        heatmap = self._create_heatmap(self.object_list)
        heatmap += self._create_heatmap(self.missing_object_list)
        if heatmap.sum() > 0:
            heatmap /= np.sum(heatmap)

        # Normalize the heatmap to [0, 1]
        heatmap /= np.max(heatmap)

        self._publish_heatmap(heatmap * 100)

    def _get_object_relevancy(
        prior_clip_feature: torch.tensor,
        objects: ProbabilisticMapObjectList,
    ) -> torch.tensor:
        """
        Get the similar objects based on the CLIP feature
        """
        if len(objects) > 0:
            objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
            objects_clip_fts = objects_clip_fts.to("cuda")

            similarity_scores = F.cosine_similarity(prior_clip_feature, objects_clip_fts, dim=1)
            similarity_scores = similarity_scores.cpu().numpy()
        else:
            similarity_scores = np.empty((0))
        return similarity_scores

    def _create_heatmap(self, object_list: ProbabilisticMapObjectList, similarity_threshold: float = 0.2):
        similarity_scores = HeatmapProvider._get_object_relevancy(self.query_feature_dev, object_list)

        heatmap = np.zeros((int(self.occupancy_info['height'] / self.occupancy_info['resolution']), int(self.occupancy_info['width'] / self.occupancy_info['resolution'])))

        kernel = self._gaussian_kernel(0.2)

        for obj, sim in zip(object_list, similarity_scores):
            if sim > similarity_threshold:
                centroids = np.vstack(obj['centroid_locations'])[:,:2]
                centroids = world_to_cell(centroids, self.occupancy_info["origin"], self.occupancy_info["resolution"])

                # remove centroids outside the grid
                centroids = centroids[(centroids[:, 0] >= 0) & (centroids[:, 0] < heatmap.shape[1]) & (centroids[:, 1] >= 0) & (centroids[:, 1] < heatmap.shape[0])]

                obj_heatmap = np.zeros(heatmap.shape)
                unique_centroids, counts_centroids = np.unique(centroids, axis=0, return_counts=True)
                obj_heatmap[unique_centroids[:, 1], unique_centroids[:, 0]] = counts_centroids

                heatmap += obj_heatmap * sim

        heatmap = F.conv2d(
            torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            padding='same'
        ).squeeze().numpy()
        if heatmap.sum() > 0:
            heatmap /= np.sum(heatmap)
            
        return heatmap

    def _publish_heatmap(self, heatmap: np.ndarray):
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid.header.frame_id = "map"
        occupancy_grid.info.resolution = self.occupancy_info["resolution"]
        occupancy_grid.info.height = heatmap.shape[0]
        occupancy_grid.info.width = heatmap.shape[1]

        occupancy_grid.info.origin = Pose()
        occupancy_grid.info.origin.position.x = float(self.occupancy_info["origin"][0])
        occupancy_grid.info.origin.position.y = float(self.occupancy_info["origin"][1])
        occupancy_grid.info.origin.position.z = 0.0
        occupancy_grid.info.origin.orientation.x = 0.0
        occupancy_grid.info.origin.orientation.y = 0.0
        occupancy_grid.info.origin.orientation.z = 0.0
        occupancy_grid.info.origin.orientation.w = 1.0

        occupancy_grid.data = heatmap.flatten(order="C").astype(np.int8).tolist()
        self.heatmap_publisher.publish(occupancy_grid)

    def _gaussian_kernel(self, kernel_size_meters: int):
        # Apply a Gaussian kernel to smooth the heatmap
        kernel_size = int(kernel_size_meters / self.occupancy_info["resolution"])

        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        sigma = kernel_size / 6.0  # Approximation for Gaussian kernel

        extent = int((kernel_size - 1) / 2)
        x = np.arange(-extent, extent + 1)
        y = np.arange(-extent, extent + 1)
        x, y = np.meshgrid(x, y)
        gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()
        return gaussian_kernel