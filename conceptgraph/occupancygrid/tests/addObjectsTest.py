from conceptgraph.occupancygrid.utils import add_objects_to_occupancy_grid, show_occupancy_grid, dilate_map
from conceptgraph.slam.utils_no_sampling import ProbabilisticMapObjectList
import open3d as o3d
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

class OccupancySubscriber(Node):
    def __init__(self):
        super().__init__('occupancy_subscriber')
        self.subscription = self.create_subscription(OccupancyGrid, 'map', self.occupancy_callback, 10)
        self.map = None
        self.map_info = None

        self.timer = self.create_timer(5, self.timer_callback)

        self.objects = ProbabilisticMapObjectList()
        self.objects.append(
            {'bbox': o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, 0, 0), max_bound=(1, 1, 1))}
            )
        self.objects.append(
            {'bbox': o3d.geometry.AxisAlignedBoundingBox(min_bound=(1, 1, 5), max_bound=(2, 2, 6))}
            )

    def occupancy_callback(self, msg):
        self.map = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_info = {
            'resolution': msg.info.resolution,
            'origin': (msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.position.z),
            'width': msg.info.width,
            'height': msg.info.height
        }

    def timer_callback(self):
        if self.map is not None:
            occupancy_grid = add_objects_to_occupancy_grid(self.map, self.map_info, self.objects, 2.0)
            show_occupancy_grid(occupancy_grid)
            grid = dilate_map(occupancy_grid, self.map_info, 0.15)
            show_occupancy_grid(grid)
    

def main():
    rclpy.init()
    occupancy_subscriber = OccupancySubscriber()
    rclpy.spin(occupancy_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()