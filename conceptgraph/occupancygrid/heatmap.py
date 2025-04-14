from conceptgraph.slam.utils_no_sampling import ProbabilisticMapObjectList
from conceptgraph.occupancygrid.utils import convert_world_to_cell
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
from torch.nn import functional as F

def get_object_heatmap(map: np.ndarray, map_info: dict, query_object: dict, objects: ProbabilisticMapObjectList):
    grid = np.zeros(map.shape)
    grid = get_prior_from_object(query_object, objects)
    grid = update_heatmap_with_locations(grid, query_object)
    grid = normalize_distribution(grid)
    return grid

def get_prior_from_object(map: np.ndarray, map_info: dict, query_object: dict, objects: ProbabilisticMapObjectList) -> np.ndarray:
    other_objects = [obj for obj in objects if obj is not query_object]
    features = torch.stack([obj['clip_ft'] for obj in other_objects])
    visual_sim = F.cosine_similarity(features, query_object['clip_ft'])

    for i, obj in enumerate(objects):
        upper = obj['bbox'].get_max_bound()
        lower = obj['bbox'].get_min_bound()
        # upper = obj['pcd'].get_max_bound()
        # lower = obj['pcd'].get_min_bound()
        corners = [(lower[1], lower[0]), (upper[1], upper[0])]
        # left-bottom, right-top
        corners = [convert_world_to_cell(corner, map_info) for corner in corners]
        map[corners[0][0]:corners[1][0], corners[0][1]:corners[1][1]] = visual_sim[i].item()

    map = smoothen_distribution(map, sigma=0.3)
    map = map / np.sum(map)
    return map

def update_heatmap_with_locations(map: np.ndarray, query_object: dict) -> np.ndarray:
    """
    Update the heatmap with the locations of the query object.
    """
    upper = query_object['bbox'].get_max_bound()
    lower = query_object['bbox'].get_min_bound()
    # upper = query_object['pcd'].get_max_bound()
    # lower = query_object['pcd'].get_min_bound()
    corners = [(lower[1], lower[0]), (upper[1], upper[0])]
    # left-bottom, right-top
    corners = [convert_world_to_cell(corner, map_info) for corner in corners]
    map[corners[0][0]:corners[1][0], corners[0][1]:corners[1][1]] = 1.0
    return map


def smoothen_distribution(distribution: np.ndarray, sigma: float=0.3) -> np.ndarray:
    """
    Smoothen the distribution using a Gaussian filter.
    """
    smoothed_distribution = gaussian_filter(distribution, sigma=sigma)
    return smoothed_distribution