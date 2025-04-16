from conceptgraph.slam.utils_no_sampling import ProbabilisticMapObjectList
from conceptgraph.occupancygrid.utils import convert_world_to_cell, convert_cell_to_world
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import numpy as np
import torch
from torch.nn import functional as F

def get_object_heatmap(map: np.ndarray, map_info: dict, prior_clip_feature: np.ndarray, objects: ProbabilisticMapObjectList) -> np.ndarray:
    similar_objects, similarity_scores = get_similar_objects(prior_clip_feature, objects)
    
    grid = np.zeros(map.shape)
    # grid = get_prior_from_clip_feature(grid, map_info, similar_objects, similarity_scores)
    print("similar objects", [o['class_name'] for o in similar_objects])
    print("similarity scores", similarity_scores)
    grid = update_heatmap_with_locations(grid, map_info, similar_objects, similarity_scores)
    grid /= np.sum(grid)
    return grid

def get_similar_objects(prior_clip_feature: torch.tensor, objects: ProbabilisticMapObjectList, similarity_threshold: float=0.2) -> tuple[list,list]:
    """
    Get the similar objects based on the CLIP feature.
    """
    objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
    objects_clip_fts = objects_clip_fts.to("cuda")

    print("prior_clip_feature", prior_clip_feature.shape)
    print("objects_clip_fts", objects_clip_fts.shape)

    visual_sim = F.cosine_similarity(
        prior_clip_feature, objects_clip_fts, dim=1
    )
    similar_objects = []
    similarity_scores = []
    for i, obj in enumerate(objects):
        sim = visual_sim[i].item()
        if sim > similarity_threshold:
            similar_objects.append(obj)
            similarity_scores.append(sim)
    return similar_objects, similarity_scores

def get_prior_from_clip_feature(map: np.ndarray, map_info: dict, similar_objects: list, similarity_scores: list) -> np.ndarray:
    """
    Get the prior from the CLIP feature.
    """
    for obj, sim in zip(similar_objects, similarity_scores):
        upper = obj['bbox'].get_max_bound()
        lower = obj['bbox'].get_min_bound()
        # upper = obj['pcd'].get_max_bound()
        # lower = obj['pcd'].get_min_bound()
        corners = [(lower[1], lower[0]), (upper[1], upper[0])] # left-bottom, right-top
        corners = [convert_world_to_cell(corner, map_info) for corner in corners]
        map[corners[0][0]:corners[1][0], corners[0][1]:corners[1][1]] = sim

    map = smoothen_distribution(map, sigma=0.3)
    map = map / np.sum(map)
    return map

def update_heatmap_with_locations(map: np.ndarray, map_info: dict, similar_objects: list, similarity_scores: list) -> np.ndarray:
    """
    Update heatmap with the centroid location of similar objects
    """
    for obj, sim in zip(similar_objects, similarity_scores):
        centroids = np.vstack(obj['centroid_locations'])[:,:2].T
        estimated_density_function = gaussian_kde(centroids)
        Y, X = np.mgrid[0:map.shape[0], 0:map.shape[1]]
        Y, X = convert_cell_to_world((Y,X), map_info)

        positions = np.vstack([Y.ravel(), X.ravel()])
        values = estimated_density_function(positions)
        values = values.reshape(map.shape)
        map += values * sim
    return map


def smoothen_distribution(distribution: np.ndarray, sigma: float=10) -> np.ndarray:
    """
    Smoothen the distribution using a Gaussian filter.
    """
    smoothed_distribution = gaussian_filter(distribution, sigma=sigma)
    return smoothed_distribution