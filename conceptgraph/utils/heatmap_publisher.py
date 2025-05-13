import numpy as np
import torch
from enum import Enum
from torch.nn import functional as F
from rclpy.node import Node
from std_msgs.msg import String as StringMsg
from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt

from conceptgraph.slam.slam_classes import ProbabilisticMapObjectList
from conceptgraph.occupancygrid.utils import world_to_cell
from geometry_msgs.msg import Pose

class SimilarityMeasure(Enum):
    COSINE = 0
    SAME_LABEL = 1

class HeatmapProvider:
    def __init__(self, clip_model, clip_tokenizer, object_list: ProbabilisticMapObjectList, missing_object_list: ProbabilisticMapObjectList, similarity_measure: SimilarityMeasure = SimilarityMeasure.SAME_LABEL, node = None):
        if node is None:
            self.node = Node("heatmap_publisher")
        else:
            self.node = node

        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.object_list = object_list
        self.missing_object_list = missing_object_list

        self.device = "cuda"
        self.similarity_measure = similarity_measure
        self.plot_heatmap = True

        self.query_text = ""
        self.query_feature_dev = None
        self.query_subscription = self.node.create_subscription(
            StringMsg, "heatmap_goal", self.query_callback, 1
        )

        self.margin = 0.2
        self.occupancy_info = {
            "resolution": 0.05,  # meters per pixel
            "width": 0.1,  # meters
            "height": 0.1,
            "origin": (0, 0),
        }
        self.lower_corner_xy = np.array(self.occupancy_info["origin"])
        self.upper_corner_xy = np.array(self.occupancy_info["origin"]) + np.array(
            (self.occupancy_info["width"], self.occupancy_info["height"])
        )
        self.heatmap_publisher = self.node.create_publisher(OccupancyGrid, "heatmap", 10)

        self.timer = self.node.create_timer(1, self.update_callback)

    def query_callback(self, msg: StringMsg) -> None:
        self.query_text = msg.data

        text_queries = [self.query_text]
        text_queries_tokenized = self.clip_tokenizer(text_queries).to("cuda")
        self.query_feature_dev = self.clip_model.encode_text(text_queries_tokenized)

        self.node.get_logger().info(f"Received query: {self.query_text}")

    def update_callback(self) -> None:
        if self.query_text == "":
            return

        # Create a heatmap from the object list
        self._update_map_size()
        heatmap = self._create_heatmap(self.object_list)
        heatmap += self._create_heatmap(self.missing_object_list)
        if heatmap.sum() > 0:
            heatmap /= np.sum(heatmap)

        self._publish_heatmap(heatmap * 255)
        if self.plot_heatmap:
            self._plot_heatmap(heatmap)

    def _update_map_size(
        self,
    ):
        if len(self.object_list) == 0:
            return
        points = np.vstack(
            [np.vstack(obj["centroid_locations"])[:, :2] for obj in self.object_list]
        )
        point_bounds_min = np.min(points, axis=0)
        point_bounds_max = np.max(points, axis=0)
        if np.any(point_bounds_min < self.lower_corner_xy - self.margin) or np.any(
            point_bounds_max > self.upper_corner_xy + self.margin
        ):
            self.lower_corner_xy = np.minimum(
                point_bounds_min, self.lower_corner_xy - self.margin
            )
            self.upper_corner_xy = np.maximum(
                point_bounds_max, self.upper_corner_xy + self.margin
            )
            self.occupancy_info["origin"] = self.lower_corner_xy    
            self.occupancy_info["width"], self.occupancy_info["height"] = np.floor((
                self.upper_corner_xy - self.lower_corner_xy
            ) / self.occupancy_info["resolution"]) * self.occupancy_info["resolution"]

    def _get_object_relevancy(
        self,
        prior_clip_feature: torch.tensor,
        objects: ProbabilisticMapObjectList,
    ) -> torch.tensor:
        """
        Get the similar objects based on the CLIP feature
        """
        if len(objects) == 0:
            return np.empty((0))

        if self.similarity_measure == SimilarityMeasure.SAME_LABEL:
            similarity_scores = np.zeros(len(objects))
            mask = [o["class_name"] == self.query_text for o in objects]
            similarity_scores[mask] = 1

        elif self.similarity_measure == SimilarityMeasure.COSINE:
            objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
            objects_clip_fts = objects_clip_fts.to("cuda")
            similarity_scores = F.cosine_similarity(prior_clip_feature, objects_clip_fts, dim=1)
            similarity_scores = similarity_scores.cpu().numpy()

        else:
            raise ValueError("Invalid similarity measure")
        
        return similarity_scores

    def _create_heatmap(self, object_list: ProbabilisticMapObjectList, similarity_threshold: float = 0.3) -> np.ndarray:
        similarity_scores = self._get_object_relevancy(self.query_feature_dev, object_list)

        heatmap = np.zeros((int(self.occupancy_info['height'] / self.occupancy_info['resolution']), int(self.occupancy_info['width'] / self.occupancy_info['resolution'])))

        kernel = self._gaussian_kernel(0.4)

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

    def _publish_heatmap(self, heatmap: np.ndarray) -> None:
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.stamp = self.node.get_clock().now().to_msg()
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
        self.node.get_logger().info("Published heatmap")

    def _gaussian_kernel(self, kernel_size_meters: int) -> np.ndarray:
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
    
    def _init_heatmap_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Heatmap")
        self.ax.set_xlabel("X (cells)")
        self.ax.set_ylabel("Y (cells)")
    
    def _plot_heatmap(self, heatmap: np.ndarray):
        if not self.plot_heatmap:
            return

        if not hasattr(self, 'fig'):
            self._init_heatmap_plot()

        self.ax.clear()
        self.ax.imshow(heatmap, cmap='hot', interpolation='nearest')
        self.ax.yaxis.set_inverted(False)
        self.ax.set_title("Heatmap")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        
        x_ticks = np.arange(0, heatmap.shape[1], step=int(heatmap.shape[1] / 5) + 1)
        y_ticks = np.arange(0, heatmap.shape[0], step=int(heatmap.shape[0] / 5) + 1)
        x_labels = (x_ticks * self.occupancy_info["resolution"] + self.occupancy_info["origin"][0]).round(2)
        y_labels = (y_ticks * self.occupancy_info["resolution"] + self.occupancy_info["origin"][1]).round(2)
        self.ax.set_xticks(x_ticks)
        self.ax.set_xticklabels(x_labels)
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(y_labels)

        plt.pause(0.01)
