import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
from scipy.ndimage import gaussian_filter, binary_dilation
import cv2
from conceptgraph.slam.utils_no_sampling import ProbabilisticMapObjectList
from enum import Enum

from copy import deepcopy

"""
All coordinates use (y,x) format for coordinates.
"""

class OccupancyGridValue(Enum):
    """
    Enum for occupancy grid values.
    """
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100

def add_objects_to_occupancy_grid(occupancy_grid: np.ndarray, occupancy_info: dict, objects: ProbabilisticMapObjectList, max_height: float) -> np.ndarray:
    """
    Add objects to an occupancy grid. Assumes objects are within the bounds of the occupancy grid.
    """
    grid = deepcopy(occupancy_grid)
    mask = np.zeros(grid.shape, dtype=bool)
    for obj in objects:
        upper = obj['bbox'].get_max_bound()
        lower = obj['bbox'].get_min_bound()
        # upper = obj['pcd'].get_max_bound()
        # lower = obj['pcd'].get_min_bound()
        if lower[2] > max_height:
            # Ignore objects above the robot
            continue
        corners = [(lower[1], lower[0]), (upper[1], upper[0])]
        # left-bottom, right-top
        corners = [convert_world_to_cell(corner, occupancy_info) for corner in corners]
        mask[corners[0][0]:corners[1][0], corners[0][1]:corners[1][1]] = True

    grid[mask] = OccupancyGridValue.OCCUPIED.value

    return grid

def convert_world_to_cell(world: tuple, occupancy_info: dict) -> tuple:
    """
    Convert world coordinates to a cell.
    """
    resolution = occupancy_info['resolution']
    origin = occupancy_info['origin']
    y = int((world[0] - origin[1]) / resolution) # Coords are (y,x) but origin is (x,y,z)
    x = int((world[1] - origin[0]) / resolution)
    assert 0 <= x < occupancy_info['width'], "Out of Bounds: Failed to convert world coordinate to grid coordinate. x: %d, width: %d" % (x, occupancy_info['width'])
    assert 0 <= y < occupancy_info['height'], "Out of Bounds: Failed to convert world coordinate to grid coordinate. y: %d, height: %d" % (y, occupancy_info['height'])
    return (y, x)

def world_to_cell(world_xy : np.ndarray, occupancy_origin_xy: tuple, occupancy_resolution: float) -> np.ndarray:
    """
    Convert world coordinates to cell coordinates.
    """
    if len(np.shape(world_xy)) == 1:
        world_xy = np.reshape(world_xy, (1, -1))
    elif not (len(np.shape(world_xy)) == 2 and np.shape(world_xy)[1] == 2):
        raise ValueError("world_xy must be a 2D array with shape (N, 2) or a 1D array with shape (2,)")
    offset_xy = world_xy - np.reshape(occupancy_origin_xy, (1, 2))
    offset_xy = np.floor(offset_xy / occupancy_resolution).astype(np.int32)
    return offset_xy

def show_occupancy_grid(grid: np.ndarray, cmap='viridis'):
    """
    Display the occupancy grid.
    """

    # print("OCCUPIED")
    # print(np.sum(grid == OccupancyGridValue.OCCUPIED.value))
    # print("FREE")
    # print(np.sum(grid == OccupancyGridValue.FREE.value))
    # print("UNKNOWN")
    # print(np.sum(grid == OccupancyGridValue.UNKNOWN.value))
    plt.imshow(grid, cmap=cmap)
    plt.gca().yaxis.set_inverted(False)
    plt.show()

def dilate_map(occupancy_grid: np.ndarray, occupancy_info: dict, dilatation_amount_metres: float) -> np.ndarray:
    """
    Dilate the occupancy grid.
    """
    grid = deepcopy(occupancy_grid)
    dilation_amount_cell = int(dilatation_amount_metres / occupancy_info['resolution'])
    h, w = grid.shape
    for i in range(h):
        for j in range(w):
            if occupancy_grid[i, j] == OccupancyGridValue.OCCUPIED.value:
                grid[max(0, i - dilation_amount_cell): min(h - 1, i + dilation_amount_cell), max(0, j - dilation_amount_cell): min(w - 1, j + dilation_amount_cell)] = OccupancyGridValue.OCCUPIED.value
    return grid

# FROM AVL MAPS Paper - here if needed
def dilate_map_avl(binary_map: np.ndarray, dilate_iter: int = 0, gaussian_sigma: float = 1.0):
    h, w = binary_map.shape
    binary_map = cv2.resize(binary_map.astype(float), (w * 2, h * 2))
    binary_map = gaussian_filter((binary_map).astype(float), sigma=gaussian_sigma, truncate=3)
    binary_map = (binary_map > 0.5).astype(np.uint8)
    binary_map = binary_dilation(
        binary_map,
        structure=np.ones((3, 3)),
        iterations=dilate_iter * 2,
    )
    binary_map = cv2.resize(binary_map.astype(float), (w, h))
    return binary_map

def adjust_map_size(map: np.ndarray, map_resolution: float, map_origin_xz: tuple, lower_corner_xy: np.ndarray, upper_corner_xy: np.ndarray):
    if lower_corner_xy[0] > upper_corner_xy[0] or lower_corner_xy[1] > upper_corner_xy[1]:
        raise ValueError("Lower corner must be less than upper corner")

    # calculate cell indices for the corners
    new_lower_cell_world, new_upper_cell_world = world_to_cell(
        np.vstack((lower_corner_xy, upper_corner_xy)),
        (0, 0),
        map_resolution,
    )

    # create new map with adjusted size
    new_height = new_upper_cell_world[1] - new_lower_cell_world[1] + 1
    new_width = new_upper_cell_world[0] - new_lower_cell_world[0] + 1
    new_origin = new_lower_cell_world * map_resolution
    new_occupancy = np.full(
        (new_height, new_width),
        -1,
        dtype=np.int8,
    )

    # insert data from the old map into the new map
    # get lower-left and upper-right corners of the old map in cell coordinates (relative to world, i.e., (0,0))
    # add half of resolution to ensure consistent rounding
    old_lower_cell_world = world_to_cell(np.array(map_origin_xz) + map_resolution/2, (0, 0), map_resolution)[0]
    old_upper_cell_world = (map.shape[1] - 1) + old_lower_cell_world[0], (map.shape[0] - 1) + old_lower_cell_world[1]

    # calculate the overlap between the old and new maps (in world cell coordinates)
    overlap_lower_cell_world = np.maximum(old_lower_cell_world, new_lower_cell_world)
    overlap_upper_cell_world = np.minimum(old_upper_cell_world, new_upper_cell_world)

    # convert the overlap coordinates to indices in the new map
    new_lower_index = overlap_lower_cell_world - new_lower_cell_world
    new_upper_index = overlap_upper_cell_world - new_lower_cell_world
    
    # convert the overlap coordinates to indices in the old map
    old_lower_index = overlap_lower_cell_world - old_lower_cell_world
    old_upper_index = overlap_upper_cell_world - old_lower_cell_world

    # copy the overlapping data from the old map to the new map
    new_occupancy[new_lower_index[1]:new_upper_index[1] + 1, new_lower_index[0]:new_upper_index[0] + 1] = map[
        old_lower_index[1]:old_upper_index[1] + 1, old_lower_index[0]:old_upper_index[0] + 1
    ]

    return new_occupancy, new_origin, new_width * map_resolution, new_height * map_resolution