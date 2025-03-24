import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformBroadcaster, TransformStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseToTransformNode(Node):
    def __init__(self):
        super().__init__('pose_to_transform_node')
        self.pose_sub = self.create_subscription(PoseStamped, 'orbslam/pose', self.pose_callback, 1)
        self.transform = None

        self.broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1/30, self.timer_callback)

    def timer_callback(self):
        if self.transform is not None:
            self.broadcaster.sendTransform(self.transform)
    
    def pose_callback(self, msg):
        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = 'camera_color_optical_frame'
        tf.child_frame_id = 'map'
        
        pose = np.eye(4)
        r = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        pose[:3, :3] = r.as_matrix()
        pose[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        # pose = np.linalg.inv(pose)
        p = pose[:3, 3]
        r = R.from_matrix(pose[:3, :3]).as_quat()

        tf.transform.translation.x = p[0]
        tf.transform.translation.y = p[1]
        tf.transform.translation.z = p[2]
        tf.transform.rotation.x = r[0]
        tf.transform.rotation.y = r[1]
        tf.transform.rotation.z = r[2]
        tf.transform.rotation.w = r[3]

        self.transform = tf


if __name__ == '__main__':
    rclpy.init()
    pose_to_transform_node = PoseToTransformNode()
    rclpy.spin(pose_to_transform_node)
    pose_to_transform_node.destroy_node()
    rclpy.shutdown()