import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from lsy_interfaces.srv import ConceptGraphQuery

class Test(Node):
    def __init__(self):
        super().__init__('test_service')
        self.goal_client = self.create_client(ConceptGraphQuery, 'conceptgraph_query_service')
        while not self.goal_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('Goal Client not available, waiting again')
        self.goal_req = ConceptGraphQuery.Request()
        self.goal_future = None

        self.pub_goal = self.create_publisher(Point, 'conceptgraph/goal', 10)


    

def main(args=None):
    rclpy.init(args=args)
    test = Test()

    while rclpy.ok():
        input_str = input('Enter query: ')
        test.goal_req.query = input_str
        test.get_logger().info('Sending goal')
        test.goal_future = test.goal_client.call_async(test.goal_req)

        while rclpy.ok() and not test.goal_future.done():
            rclpy.spin_once(test)

        center = test.goal_future.result().object_center
        print(center)
        print('Result received')
        test.pub_goal.publish(center)

    test.destroy_node()
    rclpy.shutdown()

main()