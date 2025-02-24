import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2

from message_filters import Subscriber, ApproximateTimeSynchronizer

CAMERA = 'INTEL' # INTEL, ORBBEC
SUPPORTED_CAMERAS = ['INTEL', 'ORBBEC']

class TestSubscriber(Node):
    def __init__(self):
        super().__init__('test_subscriber')

        if CAMERA == 'ORBBEC':
            self.sub_color = self.create_subscription(Image, 'camera/color/image_raw', self.color_callback, 10)
            self.sub_depth = self.create_subscription(Image, 'camera/depth/image_raw', self.depth_callback, 10)
            self.sub_pc = self.create_subscription(PointCloud2, 'camera/depth_registered/points', self.pc_callback, 10)
        elif CAMERA == 'INTEL':
            # self.sub_color = self.create_subscription(Image, 'camera/color/image_raw', self.color_callback, 10)
            # self.sub_depth = self.create_subscription(Image, 'camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
            # self.sub_pc = self.create_subscription(PointCloud2, 'camera/depth/color/points', self.pc_callback, 10)
            self.sub_color = Subscriber(self, Image, 'camera/color/image_raw')
            self.sub_depth = Subscriber(self, Image, 'camera/aligned_depth_to_color/image_raw')
            self.sub_pc = Subscriber(self, PointCloud2, 'camera/depth/color/points')
        else:
            raise ValueError(f'Invalid camera type: {CAMERA}. Currently supported: {SUPPORTED_CAMERAS}')
                    
        max_delay = 1/15
        self.ts = ApproximateTimeSynchronizer([self.sub_color, self.sub_depth], 1, max_delay)
        self.ts.registerCallback(self.sync_callback)

        
    def color_callback(self, msg):
        self.get_logger().info('Received RGB image')

    def depth_callback(self, msg):
        self.get_logger().info('Received DEPTH image')

    def pc_callback(self, msg):
        self.get_logger().info('Received PCD')

    def sync_callback(self, color, depth):
        self.get_logger().info('Received synchronized data')


def main(args=None):
    rclpy.init(args=args)
    test_subscriber = TestSubscriber()
    rclpy.spin(test_subscriber)
    test_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        