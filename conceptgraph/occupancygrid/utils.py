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

def convert_cell_to_world(cell: tuple, occupancy_info: dict) -> tuple:
    """
    Convert a cell to world coordinates.
    """
    resolution = occupancy_info['resolution']
    origin = occupancy_info['origin']
    return (cell[1] * resolution + origin[1], cell[0] * resolution + origin[0])

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

def show_occupancy_grid(occupancy_grid: np.ndarray, cmap='viridis'):
    """
    Display the occupancy grid.
    """
    grid = deepcopy(occupancy_grid)
    grid = np.flip(grid, axis=0)

    # print("OCCUPIED")
    # print(np.sum(grid == OccupancyGridValue.OCCUPIED.value))
    # print("FREE")
    # print(np.sum(grid == OccupancyGridValue.FREE.value))
    # print("UNKNOWN")
    # print(np.sum(grid == OccupancyGridValue.UNKNOWN.value))
    plt.imshow(grid, cmap=cmap)
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