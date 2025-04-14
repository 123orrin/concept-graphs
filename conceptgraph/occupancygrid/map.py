import numpy as np
from abc import ABC

class Map(ABC):
    def __init__(self):
        self._map = None
        self._map_info = None
    
    def set_map(self, map: np.ndarray):
        self.map = map

    def get_map(self) -> np.ndarray:
        return self.map
    
    def get_map_info(self) -> dict:
        return self.map_info
    