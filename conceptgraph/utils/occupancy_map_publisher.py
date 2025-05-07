from conceptgraph.occupancygrid.utils import adjust_map_size, world_to_cell
import rclpy
import rclpy.duration
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import numpy as np
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException
import ros2_numpy
import tf_transformations
from geometry_msgs.msg import Pose


class OccupancyMapPublisher(Node):
    """
    Node which publishes an occupancy map based on a 2D projection of a point cloud.

    All points above the robot base height are considered occupied.
    """
    def __init__(self):
        super().__init__("occupancy_map_publisher")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.floor_height = self._get_robot_base_z() + 0.05

        self.subscription = self.create_subscription(
            PointCloud2,
            "/spectacular_ai/point_cloud/local",
            self._point_cloud_callback,
            1,
        )
        self.points = None
        self.point_receive_time = None
        self.points_updated = False
        self.margin = 0.2
        self.lower_corner_xy = np.array((-0.5, -0.5))
        self.upper_corner_xy = np.array((0.5, 0.5))
        self.corners_updated = True

        self.publisher = self.create_publisher(OccupancyGrid, "/occupancy_map", 10)
        self.occupancy_info = {
            "resolution": 0.05,  # meters per pixel
            "width": (self.upper_corner_xy - self.lower_corner_xy)[0],  # meters
            "height": (self.upper_corner_xy - self.lower_corner_xy)[1],
            "origin": tuple(self.lower_corner_xy),
            "frame_id": "map",
        }
        self.occupancy = np.full(
            (
                int(self.occupancy_info["height"] / self.occupancy_info["resolution"]),
                int(self.occupancy_info["width"] / self.occupancy_info["resolution"]),
            ),
            -1,
            dtype=np.int8,
        )

        self.get_logger().info(
            f"Occupancy Map Publisher Node has been started. Using robot base height: {self.floor_height}"
        )

    def _get_robot_base_z(self):
        req_time = rclpy.time.Time()
        tf_available = False
        while not tf_available:
            self.get_logger().info("Waiting for transform from 'map' to 'base_link'")
            rclpy.spin_once(self, timeout_sec=1)
            try:
                tf_available = self.tf_buffer.can_transform(
                    "map",
                    "base_link",
                    req_time,
                    timeout=rclpy.duration.Duration(seconds=1),
                )
            except LookupException as e:
                continue
        tf = self.tf_buffer.lookup_transform("map", "base_link", req_time)
        return tf.transform.translation.z

    def _point_cloud_callback(self, msg):
        # only update when last message was used
        if not self.points_updated:
            # Convert PointCloud2 to numpy array
            self.points = ros2_numpy.point_cloud2.pointcloud2_to_xyz_array(msg).T
            self.point_receive_time = msg.header.stamp
            self.points_updated = True

            point_bounds_min = np.min(self.points[:2,:], axis=1)
            point_bounds_max = np.max(self.points[:2,:], axis=1)
            if np.any(point_bounds_min < self.lower_corner_xy - self.margin) or np.any(point_bounds_max > self.upper_corner_xy + self.margin):
                self.lower_corner_xy = np.minimum(point_bounds_min, self.lower_corner_xy - self.margin)
                self.upper_corner_xy = np.maximum(point_bounds_max, self.upper_corner_xy + self.margin)
                self.corners_updated = True

    def _ready(self):
        return self.points_updated and self.tf_buffer.can_transform(
            "map",
            "camera_color_optical_frame",
            self.point_receive_time,
            timeout=rclpy.duration.Duration(seconds=0.05),
        )

    def _publish_map(self, map: np.ndarray):
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid.header.frame_id = "map"
        occupancy_grid.info.resolution = self.occupancy_info["resolution"]
        occupancy_grid.info.height = map.shape[0]
        occupancy_grid.info.width = map.shape[1]

        occupancy_grid.info.origin = Pose()
        occupancy_grid.info.origin.position.x = float(self.occupancy_info["origin"][0])
        occupancy_grid.info.origin.position.y = float(self.occupancy_info["origin"][1])
        occupancy_grid.info.origin.position.z = 0.0
        occupancy_grid.info.origin.orientation.x = 0.0
        occupancy_grid.info.origin.orientation.y = 0.0
        occupancy_grid.info.origin.orientation.z = 0.0
        occupancy_grid.info.origin.orientation.w = 1.0

        occupancy_grid.data = map.flatten(order="C").astype(np.int8).tolist()
        self.publisher.publish(occupancy_grid)

    def _adjust_map_size(
        self, lower_corner_xy: np.ndarray, upper_corner_xy: np.ndarray
    ):
        (
            self.occupancy,
            self.occupancy_info["origin"],
            self.occupancy_info["width"],
            self.occupancy_info["height"],
        ) = adjust_map_size(
            self.occupancy,
            self.occupancy_info["resolution"],
            self.occupancy_info["origin"],
            lower_corner_xy,
            upper_corner_xy,
        )

    def main(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self._ready():
                continue
            self.points_updated = False

            tf = self.tf_buffer.lookup_transform(
                "map", "camera_color_optical_frame", self.point_receive_time
            )

            # Create homogeneous transformation matrix from tf
            rotation = tf.transform.rotation
            rotation_quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
            tf_matrix = np.eye(4)
            tf_matrix[:3, :3] = tf_transformations.quaternion_matrix(
                rotation_quaternion
            )[:3, :3]
            translation = tf.transform.translation
            tf_matrix[:3, 3] = (translation.x, translation.y, translation.z)

            points = (
                tf_matrix @ np.vstack((self.points, np.ones((1, self.points.shape[1]))))
            )[:3, :].T

            # update map size if needed
            if self.corners_updated:
                self._adjust_map_size(self.lower_corner_xy - self.margin, self.upper_corner_xy + self.margin)
                self.corners_updated = False

            cells = world_to_cell(
                points[:, :2],
                self.occupancy_info["origin"],
                self.occupancy_info["resolution"],
            )

            occupied = (points[:, 2] > self.floor_height) & (points[:, 2] < self.robot_height)
            inside_map = (cells[:, 0] >= 0) & (cells[:, 0] < self.occupancy.shape[1]) & (cells[:, 1] >= 0) & (cells[:, 1] < self.occupancy.shape[0])
            occupied_cells = cells[np.logical_and(occupied, inside_map), :]
            unoccupied_cells = cells[np.logical_and(np.logical_not(occupied), inside_map), :]

            self.occupancy[unoccupied_cells[:, 1], unoccupied_cells[:, 0]] = 0
            self.occupancy[occupied_cells[:, 1], occupied_cells[:, 0]] = 100

            self.get_logger().info("Occupancy map published.")
            self._publish_map(self.occupancy)


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMapPublisher()
    node.main()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
