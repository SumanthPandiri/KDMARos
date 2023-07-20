from functools import partial
import os, time
import torch

from env.scenarios import  rosScenario
from models.networks import ExpertNetwork
from models.env import Env
from models.agent import DLAgent

import config

import rclpy
import pubsub
from pubsub import kdmaPublisher, rosbagSubscriber, goalSubscriber



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--max_trials", type=int, default=50)
parser.add_argument("--num_neighbors", type=int, default=1)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--visualize", type=bool, default=False)
settings = parser.parse_args()


def evaluate(ckpt_file):
    #Load in the model
    print(ckpt_file)
    #print(settings.scene)

    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = {}
    for k, v in ckpt["model"].items():
        if "model.actor.model." in k:
            state_dict[k[18:]] = v
    model = ExpertNetwork(agent_dim=3, neighbor_dim=4, out_dim=2)
    model.load_state_dict(state_dict)
    if settings.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device
    model.to(device)

    
    #Set up the publisher
    rclpy.init()
    kdma_Publisher = kdmaPublisher()

    #Set up the subscribers for the neighbors and robot, and get initial positions (first is robot)
    numNeighbors = settings.num_neighbors
    rosbag_subscriber = []
    # posVel = []
    for x in range(numNeighbors):
        rosbag_subscriber.append(rosbagSubscriber('odometry/controller_' + str(x+1)))
    #     rclpy.spin_once(rosbag_subscriber[x])
    #     posVel.append([rosbag_subscriber[x].posx, rosbag_subscriber[x].posy, \
    #                 rosbag_subscriber[x].velx, rosbag_subscriber[x].vely])

    #Set up a goal subscriber and get the goal published
    # goal_Subscriber = goal_Subscriber('/move_base_simple/goal')
    # rclpy.spin_once(goal_Subscriber)
    # goalx, goaly = goal_Subscriber.posx, goal_Subscriber.posy

    goalList = [2.,2.]
    posVel = [[2.,0.],[3.,3.],[0.,2.]]

    agent_wrapper = partial(DLAgent,
        preferred_speed=config.PREFERRED_SPEED, max_speed=config.MAX_SPEED,
        observe_radius=config.NEIGHBORHOOD_RADIUS,
        expert=None, model=model
        )
    scenario = rosScenario(n_agents=numNeighbors+1, agent_wrapper=agent_wrapper, \
                           agent_goal=goalList, pos_vel=posVel)
    
    env = Env(scenario=scenario, fps=1./config.STEP_TIME, timeout=config.VISUALIZATION_TIMEOUT, \
            frame_skip=config.FRAME_SKIP, view=settings.visualize, numNeighbors = scenario.n_agents-1
    )

    model.eval()
    

    done, info = True, None
    trials = 0
    while True:
        if done:
            state = env.reset()
            if settings.visualize:
                env.figure.axes.set_title(os.path.join(os.path.basename(os.path.dirname(ckpt_file)), os.path.basename(ckpt_file)))
            t = time.time()
        else:
            state = state_


        act = []
        for ag, s in zip(env.agents,state):
            if ag in env.info["neighbors"]:
                act.append((0,0))
                continue
            #rclpy.spin_once(goal_Subscriber)
            #goalList = [goal_Subscriber.posx, goal_Subscriber.posy]
            #print(ag.goal)

            #ag.goal = 0, 0
            act.append(ag.act(s,env))

        linvel, angvel = act[0]
        print(act)
        kdma_Publisher.publishVel(linvel, angvel)

        n = []
        for x in range(numNeighbors ):
            rclpy.spin_once(rosbag_subscriber[x])
            n.append([rosbag_subscriber[x].posx, rosbag_subscriber[x].posy, \
                      rosbag_subscriber[x].velx, rosbag_subscriber[x].vely])


        state_, rews, done, info = env.step(act, n)
        delay = config.STEP_TIME - time.time() + t
        if delay > 0:
            time.sleep(delay)
        t = time.time()
        if done:
            trials += 1
            time.sleep(2)
            if trials >= settings.max_trials:
                kdma_Publisher.destroy_node()
                rclpy.shutdown()
                break


if __name__ == "__main__":

    if os.path.isfile(settings.ckpt):
        evaluate(settings.ckpt)
    else:
        def check(path):
            for f in sorted(os.listdir(path)):
                filename = os.path.join(path, f)
                if "ckpt" == f:
                    evaluate(filename)
                elif os.path.isdir(filename):
                    check(filename)
        check(settings.ckpt)

    
