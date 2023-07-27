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
from pubsub import robotPub

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

    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = {}
    for k, v in ckpt["model"].items():
        if "model.actor.model." in k:
            state_dict[k[18:]] = v
    model = ExpertNetwork(agent_dim=4, neighbor_dim=5, out_dim=2)
    model.load_state_dict(state_dict)
    if settings.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device
    model.to(device)

    if settings.num_neighbors is None:
        settings.num_neighbors = 0

    #Set up the publisher
    rclpy.init()
    
    robot_Pub = robotPub(model, settings, ckpt_file)

    rclpy.spin(robot_Pub)

    robot_Pub.destroy_node()
    rclpy.shutdown()


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

    
