from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from env.scenarios import  rosScenario
from models.networks import ExpertNetwork
from models.env import Env
from models.agent import DLAgent
import config

import numpy as np
from functools import partial
import os
import math

import matplotlib.pyplot as plt

def euler_from_quaternion(quaternion):

    #This is from https://gist.github.com/salmagro/2e698ad4fbf9dae40244769c5ab74434
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    q = [x,y,z,w]
    yaw = math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),1 - 2 * (q[1] * q[1] + q[2] * q[2]))
    roll = math.asin(2.0 * (q[0] * q[2] - q[3] * q[1]))
    pitch = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))

    return yaw



class robotPub(Node):

    def __init__(self, model, settings, ckpt_file):
        super().__init__('robotPub')
        self.model = model
        self.envCreated=False
        self.ckpt_file=ckpt_file
        self.numNeighbors = settings.num_neighbors
        self.settings = settings
        self.goal = []
        self.goalRec = False
        self.goalSub = self.create_subscription(PoseStamped,'/move_base_simple/goal', self.goal_cb, 10)
        self.goalSub
        self.done=False
        self.heading = []


        self.subTopicList = ['odometry/controller_1']
        for i in range(1,self.numNeighbors+1):
            self.subTopicList.append('odometry/tracker_' + str(i))

        self.lambdas = []
        self.subInfos = []
        self.info = []
        self.infoRec = []
        for i in range(len(self.subTopicList)):
            self.info.append([0]*7)
            self.infoRec.append(False)
            self.lambdas.append(partial(self.getInfo, i))
            self.subInfos.append(self.create_subscription(Odometry, \
                                                          self.subTopicList[i], self.lambdas[i],10))

        #Create the publisher
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.01 # seconds
        self.timer = self.create_timer(timer_period, self.actPub)
        self.i = 0



    def goal_cb(self, msg):
        self.goal = [msg.pose.position.x, msg.pose.position.y]
        self.get_logger().info('I heard: "%s"' % str(self.goal))
        self.goalRec = True


    def getInfo(self, i, msg):
        #print(msg.pose)
        posx = msg.pose.pose.position.x
        posy = msg.pose.pose.position.y
        velx = msg.twist.twist.linear.x
        vely = msg.twist.twist.linear.y
        arr = msg.pose.pose.orientation
        ori = euler_from_quaternion(arr)
        if i == 0 and self.goalRec == True:
            self.info[i] = [posx, posy, velx, vely, ori, self.goal[0], self.goal[1]]
            self.infoRec[i] = True

        elif i==0 and self.goalRec == False:
            self.infoRec[i] = False
        else:
            self.info[i] = [posx, posy, velx, vely, ori]
            self.infoRec[i] = True

        #self.get_logger().info('I heard: "%s"' % str(i))



    def actPub(self):
        if not self.goalRec:
            print('No Goal Recieved')
            return
        for i in range(len(self.infoRec)):
            if self.infoRec[i] == False:
                print('Missing readings from ' + self.subTopicList[i])
                return
            
        #create the environnment
        if self.envCreated == False:
            agent_wrapper = partial(DLAgent,
                preferred_speed=config.PREFERRED_SPEED, max_speed=config.MAX_SPEED,
                observe_radius=config.NEIGHBORHOOD_RADIUS,
                expert=None, model=self.model)
            scenario = rosScenario(n_agents=self.numNeighbors+1,\
                                    agent_wrapper=agent_wrapper, info=self.info)
            
            self.env = Env(scenario=scenario, fps=1./config.STEP_TIME, \
                           timeout=config.VISUALIZATION_TIMEOUT, frame_skip=config.FRAME_SKIP, \
                           view=self.settings.visualize, numNeighbors = self.numNeighbors)

            self.model.eval()
            self.envCreated = True
            self.state = self.env.reset()


            if self.settings.visualize:
                self.env.figure.axes.set_title(os.path.join(os.path.basename(os.path.dirname(self.ckpt_file)), \
            os.path.basename(self.ckpt_file)))

        
        act = []
        for ag, s in zip(self.env.agents,self.state):
            if ag in self.env.info["neighbors"]:
                act.append((0,0))
                continue

            act.append(ag.act(s,self.env))

        linvel, angvel = act[0]
        

        msg = Twist()
        msg.linear.x = linvel
        msg.angular.z = angvel
        self.publisher.publish(msg)
        #print(act)
        msg = [linvel, angvel]
        self.get_logger().info('Publishing: "%s"' % msg)
        self.i += 1

        print(self.info)
        self.state, rews, self.done, info = self.env.step(act, self.info)
        if self.done:
            print("GOAL HAS BEEN REACHED")

            self.goalRec = False
            self.envCreated = False
            return

        





























class rosbagSubscriber(Node):
    def __init__(self, topicname):
        super().__init__('rosbag_subscriber')
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

class goalSubscriber(Node):
    def __init__(self, topicname):
        super().__init__('goal_subscriber')
        self.subscription = self.create_subscription(
            PoseStamped,
            topicname,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.posx = 0
        self.posy = 0
        

    def listener_callback(self, msg):
        self.posx = msg.pose.position.x
        self.posy = msg.pose.position.y
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

