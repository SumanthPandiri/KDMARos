from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class rosbagSubscriber(Node):
    def __init__(self, topicname):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            topicname,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.posx = 0
        self.posy = 0
        self.velx = 0
        self.vely = 0

    def listener_callback(self, msg):
        self.posx = msg.pose.pose.position.x
        self.posy = msg.pose.pose.position.y
        self.velx = msg.twist.twist.linear.x
        self.vely = msg.twist.twist.linear.y
        #self.get_logger().info('I heard: "%s"' % str(self.vely))


class kdmaPublisher(Node):
    def __init__(self):
        super().__init__('kdma_Publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.i = 0
    
    def publishVel(self, l, a):
        #Format: {linear: {x: l, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: a}}
        msg = Twist()
        msg.linear.x = l
        msg.angular.z = a
        self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s" ' % msg.linear.x)