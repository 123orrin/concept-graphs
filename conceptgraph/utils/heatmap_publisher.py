import numpy as np
import torch
from enum import Enum
from torch.nn import functional as F
from rclpy.node import Node
from std_msgs.msg import String as StringMsg
from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from conceptgraph.slam.slam_classes import ProbabilisticMapObjectList
from conceptgraph.occupancygrid.utils import world_to_cell
from geometry_msgs.msg import Pose
from conceptgraph.llms.similarity_parallel_prompting import SimilarityOpenAIAsyncClient
from cv2 import fillPoly

class SimilarityMeasure(Enum):
    COSINE = 0
    SAME_LABEL = 1
    SEMANTIC_LLM = 2

class HeatmapProvider:
    def __init__(self, clip_model, clip_tokenizer, object_list: ProbabilisticMapObjectList, missing_object_list: ProbabilisticMapObjectList, similarity_measure: SimilarityMeasure = SimilarityMeasure.SAME_LABEL, node = None, obj_classes: list = None):
        if node is None:
            self.node = Node("heatmap_publisher")
        else:
            self.node = node

        self.use_bounding_boxes = True
        if similarity_measure == SimilarityMeasure.SEMANTIC_LLM and obj_classes is None:
            raise ValueError("Object classes must be provided when using LLM similarity.")
        if similarity_measure == SimilarityMeasure.SEMANTIC_LLM:
            self.llm_client = SimilarityOpenAIAsyncClient(obj_classes)
            self.obj_classes = obj_classes
            self.obj_classes_similarity = {obj_classes[i]: 0 for i in range(len(obj_classes))}
        elif similarity_measure == SimilarityMeasure.COSINE:
            self.device = "cuda"
            self.query_feature_dev = None
            self.clip_model = clip_model
            self.clip_tokenizer = clip_tokenizer

        self.object_list = object_list
        self.missing_object_list = missing_object_list

        self.similarity_measure = similarity_measure
        self.plot_heatmap = True

        self.query_text = ""
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
        if msg.data == self.query_text:
            self.node.get_logger().info(f"Received unchanged query: {self.query_text}")
            return

        self.query_text = msg.data

        self.node.get_logger().info(f"Received new query: {self.query_text}")
        if self.similarity_measure == SimilarityMeasure.SEMANTIC_LLM:
            self._get_object_relevancy_llm(self.query_text, )
        elif self.similarity_measure == SimilarityMeasure.COSINE:
            text_queries = [self.query_text]
            text_queries_tokenized = self.clip_tokenizer(text_queries).to("cuda")
            self.query_feature_dev = self.clip_model.encode_text(text_queries_tokenized)

    def _get_object_relevancy_llm(self, text_query: str) -> np.ndarray:
        self.node.get_logger().info(f"Query semantic similarity")
        similarity = self.llm_client.query_semantic_similarity(text_query)
        self.obj_classes_similarity = {key: similarity[i] for i, key in enumerate(self.obj_classes)}

    def update_callback(self) -> None:
        if self.query_text == "":
            return

        # Create a heatmap from the object list
        self._update_map_size()
        heatmap, object_similarity_scores = self._create_heatmap_unnormalized(self.object_list)
        heatmap_missing, missing_object_similarity = self._create_heatmap_unnormalized(self.missing_object_list)
        heatmap += heatmap_missing
        if heatmap.sum() > 0:
            heatmap /= np.sum(heatmap)
            # We skip the area normalization for now as this makes the values could make
            # the values too small for integer representation (which is needed for occupancy grid)
            # area = np.size(heatmap) * self.occupancy_info["resolution"]**2
            # heatmap /= area

        self._publish_heatmap(heatmap * 255)
        if self.plot_heatmap:
            self._plot_heatmap(heatmap, object_similarity_scores, missing_object_similarity)

    def _update_map_size(
        self,
    ):
        if len(self.object_list) == 0:  
            return
        points = np.vstack(
            [np.vstack([hull for hull in obj['bbox_shadow_hull_history']]) for obj in self.object_list]
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

    def _get_object_relevancy_same_label(
            self,
            query_text: str,
            objects: ProbabilisticMapObjectList
        ) -> np.ndarray:
        if len(objects) == 0:
            return np.empty((0))

        similarity_scores = np.zeros(len(objects))
        mask = [o["class_name"] == query_text for o in objects]
        similarity_scores[mask] = 1
        return similarity_scores

    def _get_object_relevancy_cosine_similiarity(
        query_feature_dev: torch.tensor,
        objects: ProbabilisticMapObjectList,
    ) -> torch.tensor:
        """
        Get the similar objects based on the CLIP feature
        """
        if len(objects) == 0:
            return np.empty((0))

        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarity_scores = F.cosine_similarity(query_feature_dev, objects_clip_fts, dim=1)
        return  similarity_scores.cpu().numpy()

    def _create_heatmap_unnormalized(self, object_list: ProbabilisticMapObjectList, similarity_threshold: float = 0.3) -> np.ndarray:
        if self.similarity_measure == SimilarityMeasure.SEMANTIC_LLM:
            similarity_scores = np.array([self.obj_classes_similarity[obj["class_name"]] for obj in object_list])
        elif self.similarity_measure == SimilarityMeasure.SAME_LABEL:
            similarity_scores = self._get_object_relevancy_same_label(self.query_text, object_list)
        elif self.similarity_measure == SimilarityMeasure.COSINE:
            similarity_scores = self._get_object_relevancy_cosine_similiarity(self.query_feature_dev, object_list)
        else:
            raise ValueError("Invalid similarity measure")

        heatmap = np.zeros((int(self.occupancy_info['height'] / self.occupancy_info['resolution']), int(self.occupancy_info['width'] / self.occupancy_info['resolution'])))

        kernel = self._gaussian_kernel(0.4)
        for obj, sim in zip(object_list, similarity_scores):
            if sim > similarity_threshold:
                obj_heatmap = np.zeros_like(heatmap)
                if self.use_bounding_boxes:
                    for bbox_shadow_hull_history in obj['bbox_shadow_hull_history']:
                        shadow_polygon_cell = world_to_cell(bbox_shadow_hull_history, self.occupancy_info["origin"], self.occupancy_info["resolution"])
                        obj_heatmap += fillPoly(np.zeros_like(heatmap), [shadow_polygon_cell], color=1)

                else:
                    centroids = np.vstack(obj['centroid_locations'])[:,:2]
                    centroids = world_to_cell(centroids, self.occupancy_info["origin"], self.occupancy_info["resolution"])

                    # remove centroids outside the grid
                    centroids = centroids[(centroids[:, 0] >= 0) & (centroids[:, 0] < heatmap.shape[1]) & (centroids[:, 1] >= 0) & (centroids[:, 1] < heatmap.shape[0])]

                    unique_centroids, counts_centroids = np.unique(centroids, axis=0, return_counts=True)
                    obj_heatmap[unique_centroids[:, 1], unique_centroids[:, 0]] = counts_centroids

                heatmap += obj_heatmap * sim

        heatmap = F.conv2d(
            torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            padding='same'
        ).squeeze().numpy()

        return heatmap, similarity_scores

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
    
    def _plot_heatmap(self, heatmap: np.ndarray, similarity_scores: np.ndarray, similarity_scores_missing: np.ndarray) -> None:
        if not self.plot_heatmap:
            return

        if not hasattr(self, 'fig'):
            self._init_heatmap_plot()

        self.ax.clear()

        # Create a meshgrid for the heatmap and plot it
        X, Y = np.meshgrid(
            np.arange(0, heatmap.shape[1]) * self.occupancy_info["resolution"] + self.occupancy_info["origin"][0],
            np.arange(0, heatmap.shape[0]) * self.occupancy_info["resolution"] + self.occupancy_info["origin"][1],
        )
        self.ax.pcolor(X, Y, heatmap, cmap='hot')

        # Normalize the similarity scores to [0, 1]
        sims = np.concatenate([similarity_scores, similarity_scores_missing])
        if len(sims) == 0 or np.max(sims) == 0:
            max_sim = 1
        else:
            max_sim = np.max(sims)

        # Project the point clouds to 2D and plot them
        cmap = cm.get_cmap('coolwarm')
        for obj, sim in zip(self.object_list, similarity_scores):
            points = np.asarray(obj['pcd'].points)[:, :2]
            norm_sim = sim / max_sim
            color = cmap(norm_sim)
            self.ax.scatter(points[:, 0], points[:, 1], s=2, color=[color], alpha=0.5)
        
        for obj, sim in zip(self.missing_object_list, similarity_scores_missing):
            points = np.asarray(obj['pcd'].points)[:, :2]
            norm_sim = sim / max_sim
            color = cmap(norm_sim)
            self.ax.scatter(points[:, 0], points[:, 1], s=2, color=[color], alpha=0.5)

        # Plot the heatmap again on top
        self.ax.pcolor(X, Y, heatmap, cmap='hot', alpha=0.3)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Heatmap for query: " + self.query_text)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

        plt.pause(0.01)
