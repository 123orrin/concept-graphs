import rclpy
from rclpy.node import Node
import time

from sensor_msgs.msg import Image

class test(Node):
	def __init__(self):
		super().__init__('test')
		self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw/compressed', self.rgb_cb, 1)
		self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw/compressed', self.depth_cb, 1)
		self.rgb_sub
		self.depth_sub

	def rgb_cb(self, msg):
		print('GOT RGB IMAGE')
	
	def depth_cb(self, msg):
		print('GOT DEPTH IMAGE')
	
if __name__ == '__main__':
	rclpy.init()
	node = test()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()