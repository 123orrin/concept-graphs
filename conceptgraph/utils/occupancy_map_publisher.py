from conceptgraph.occupancygrid.utils import adjust_map_size, world_to_cell
import rclpy
import rclpy.duration
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import OccupancyGrid
import numpy as np
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException
import ros2_numpy
import tf_transformations
from geometry_msgs.msg import Pose
import torch
from sensor_msgs.msg import PointCloud2
import ros2_numpy.point_cloud2 as point_cloud2

from conceptgraph.utils.voxel import VoxelizedPointcloud

from scipy.spatial import cKDTree

import open3d as o3d
import numpy as np


visualize_voxel = False



class OccupancyMapPublisher():
    """
    Node which publishes an occupancy map based on a 2D projection of a point cloud.

    All points above the robot base height are considered occupied.
    """
    def __init__(self, node = None):
        if node is None:
            self.node = Node("occupancy_map_publisher")
        else:
            self.node = node

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self.floor_height = self._get_robot_base_z() + 0.05
        self.robot_height = self.floor_height + 1.8

        self.points = None
        self.point_receive_time = None
        self.points_updated = False
        self.margin = 0.2
        self.lower_corner_xy = np.array((-0.5, -0.5))
        self.upper_corner_xy = np.array((0.5, 0.5))
        self.corners_updated = True
        self.points = []
        self.object_list = None
        self.background_pc = None

        self.objects = None
        self.objects_recive_time = None
        self.has_recived_objects = False

        self.global_map = None

        self.publisher = self.node.create_publisher(OccupancyGrid, "/occupancy_map", 10)
        self.global_pointcloud_publisher = self.node.create_publisher(
            PointCloud2, "/global_pointcloud", 10
        )
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
        self.node.get_logger().info(
            f"Occupancy Map Publisher Node has been started. Using robot base height: {self.floor_height}"
        )

    def _publish_global_pc(self):
        """       
        Publishes the global voxelized point cloud as a PointCloud2 message.
        """

        points = self.global_vox.get_pointcloud()[0].cpu().numpy()  # Shape: (N, 3)
        # Ensure the points array has the correct structured dtype
        structured_points = np.zeros(points.shape[0], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ])
        structured_points['x'] = points[:, 0]
        structured_points['y'] = points[:, 1]
        structured_points['z'] = points[:, 2]
        # Create a PointCloud2 message
        pointcloud_msg = ros2_numpy.msgify(PointCloud2, structured_points, frame_id="map")
        # Publish the PointCloud2 message
        self.global_pointcloud_publisher.publish(pointcloud_msg)


    def _get_robot_base_z(self):
        return -1.5
        req_time = rclpy.time.Time()
        tf_available = False
        while not tf_available:
            self.node.get_logger().info("Waiting for transform from 'map' to 'base_link'")
            rclpy.spin_once(self.node, timeout_sec=1)
            try:
                tf_available = self.node.tf_buffer.can_transform(
                    "map",
                    "base_link",
                    req_time,
                    timeout=rclpy.duration.Duration(seconds=1),
                )
            except LookupException as e:
                continue
        tf = self.node.tf_buffer.lookup_transform("map", "base_link", req_time)
        return tf.transform.translation.z

    def _point_cloud_callback(self, msg, color_msg=None):
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

    def _object_callback(self,msg):
        # Convert PointCloud2 to numpy array
        self.objects = ros2_numpy.point_cloud2.pointcloud2_to_xyz_array(msg).T
        self.objects_recive_time = msg.header.stamp
        self.has_recived_objects= True

        point_bounds_min = np.min(self.objects[:2,:], axis=1)
        point_bounds_max = np.max(self.objects[:2,:], axis=1)
        if np.any(point_bounds_min < self.lower_corner_xy - self.margin) or np.any(point_bounds_max > self.upper_corner_xy + self.margin):
            self.lower_corner_xy = np.minimum(point_bounds_min, self.lower_corner_xy - self.margin)
            self.upper_corner_xy = np.maximum(point_bounds_max, self.upper_corner_xy + self.margin)
            self.corners_updated = True


    def _map_callback(self, msg):
        self.global_map = ros2_numpy.point_cloud2.pointcloud2_to_xyz_array(msg).T

    def _ready(self):
        if not self.points_updated:
            return False
        else:
            t =self.tf_buffer.can_transform(
                    "map",
                    "camera_color_optical_frame",
                    self.point_receive_time,
                    timeout=rclpy.duration.Duration(seconds=0.2),
                    return_debug_tuple=True
                )
            if t[0]:
                return True
            return False

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




    #def publish_occupancy_map(self):
        # add backrgound point cloud to voxel map

        # tempoarily add the object point clouds to the voxel map/point cloud obtained from the voxel map/occupancy map

        # (combine with lidar)

        # publish point cloud/occupancy map




    def test_publish_pcd(self):
        all_points = []
        all_colors = []

        # Add object points
        for obj in self.object_list:
            if obj and obj["pcd"] is not None:
                points = np.asarray(obj["pcd"].points)
                if obj["pcd"].has_colors():
                    colors = np.asarray(obj["pcd"].colors)
                else:
                    colors = np.zeros_like(points)
                if len(points) > 0:
                    all_points.append(points)
                    all_colors.append(colors)

        # Add background points
        if self.background_pc is not None and len(self.background_pc.points) > 0:
            bg_points = np.asarray(self.background_pc.points)
            if self.background_pc.has_colors():
                bg_colors = np.asarray(self.background_pc.colors)
            else:
                bg_colors = np.zeros_like(bg_points)
            all_points.append(bg_points)
            all_colors.append(bg_colors)

        # Concatenate all
        if all_points:
            all_points = np.concatenate(all_points, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)
            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd.points = o3d.utility.Vector3dVector(all_points)
            combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
            pc_msg = o3dpcd_to_pointcloud2(combined_pcd, frame_id="map")
            if pc_msg is not None:
                self.global_pointcloud_publisher.publish(pc_msg)


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

            if self.has_recived_objects and self.objects is not None:
                dynamic_points = (
                    tf_matrix @ np.vstack((self.objects, np.ones((1, self.objects.shape[1]))))
                )[:3, :].T

                dtype = [('x', float), ('y', float), ('z', float)]
                structured_dynamic = np.ascontiguousarray(dynamic_points).view(dtype)
                structured_points = np.ascontiguousarray(points).view(dtype)
                static_points = structured_points[~np.isin(structured_points, structured_dynamic)]
                static_points = static_points.view(np.float64).reshape(-1, 3)

                """
                distances,_ = dynamic_tree.query(points, distance_upper_bound=1)
                static_points = points[distances == np.inf]

                static_points = np.hstack((static_points, np.zeros((static_points.shape[0],1))))
                """
                static_points = np.hstack((static_points, np.zeros((static_points.shape[0],1))))
            else:
                continue

            #static_points = np.hstack((static_points, np.zeros((static_points.shape[0],1))))
            # Convert points to a PyTorch tensor
            static_points = torch.tensor(static_points, dtype=torch.float32, device='cpu')
            self.global_vox.add(static_points,None,None)

            points = self.global_vox.get_pointcloud()[0]
            points = points.cpu().numpy()
            points = points[:, :3]
            if self.has_recived_objects and self.objects is not None:
                points = np.vstack((points, dynamic_points))


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

            self.node.get_logger().info("Occupancy map published.")
            self._publish_map(self.occupancy)

            self._publish_global_pc()
            self.node.get_logger().info("Global point cloud published.")


            if visualize_voxel:
                visualize_voxel_o3d(self.global_vox,self.global_vox.voxel_size)
            self.obj_recived = False
            points = points[:len(points)-len(dynamic_points)]

            
def visualize_voxel_o3d(voxel_grid, voxel_size):
    voxel_centers = voxel_grid.get_voxel_centers()

    voxel_mesh = o3d.geometry.TriangleMesh()
    for center in voxel_centers:
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        cube.translate(center - voxel_size /2)
        voxel_mesh += cube
    
    voxel_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([voxel_mesh])

def o3dpcd_to_pointcloud2(pcd, frame_id="map"):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        arr = np.zeros(points.shape[0], dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('r', np.float32), ('g', np.float32), ('b', np.float32)
        ])
        arr['x'] = points[:, 0]
        arr['y'] = points[:, 1]
        arr['z'] = points[:, 2]
        arr['r'] = colors[:, 0] * 255
        arr['g'] = colors[:, 1] * 255
        arr['b'] = colors[:, 2] * 255
    else:
        arr = np.zeros(points.shape[0], dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32)
        ])
        arr['x'] = points[:, 0]
        arr['y'] = points[:, 1]
        arr['z'] = points[:, 2]
    msg = point_cloud2.array_to_pointcloud2(arr, stamp=None, frame_id=frame_id)
    return msg

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMapPublisher()
    node.main()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
