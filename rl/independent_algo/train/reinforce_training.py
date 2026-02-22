from ultimatetictactoe import ultimatetictactoe
from rl.agent import NeuralAgent
from rl.independent_algo.reinforce import reinforce

import torch
from torch import nn
import torch.optim as optim

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import random
import pygame

from rl.agent import state_to_tensor

import pickle

import argparse
import torch

def parse_args():
    p = argparse.ArgumentParser()

    # training
    p.add_argument("--num_episodes", type=int, default=5000)
    p.add_argument("--enable_swap", action="store_true")
    p.add_argument("--enable_transform", action="store_true")
    p.add_argument("--px", type=float, default=0.0)
    p.add_argument("--pt", type=float, default=0.0)
    p.add_argument("--checkpoint_rate", type=int, default=100)
    p.add_argument("--validation_rate", type=int, default=100)
    
    p.add_argument("--weights_path", type=str, default="")
    p.add_argument("--weights_name", type=str, default="")

    p.add_argument("--experiment_name", type=str, default="const")
    p.add_argument("--seed", type=int, default=42)

    # shared agent params
    p.add_argument("--eps", type=float, default=0.3)

    # agent 1 params
    p.add_argument("--a1_learning_power", type=float, default=6)
    p.add_argument("--a1_learning_const", type=float, default=1.0)
    p.add_argument("--a1_exploration_power", type=float, default=2)
    p.add_argument("--a1_exploration_const", type=float, default=1.0)

    # agent 2 params
    p.add_argument("--a2_learning_power", type=float, default=10.5)
    p.add_argument("--a2_learning_const", type=float, default=1.0)
    p.add_argument("--a2_exploration_power", type=float, default=1)
    p.add_argument("--a2_exploration_const", type=float, default=1.0)

    # device
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    return p.parse_args()

args = parse_args()

env = ultimatetictactoe.env(render_mode="rgb_array")
env.reset(seed=args.seed)

a1 = NeuralAgent(
    "player_1",
    epsilon=args.eps,
    learning_power=args.a1_learning_power,
    learning_const=args.a1_learning_const,
    exploration_power=args.a1_exploration_power,
    exploration_const=args.a1_exploration_const
)
a2 = NeuralAgent(
    "player_2",
    epsilon=args.eps,
    learning_power=args.a2_learning_power,
    learning_const=args.a2_learning_const,
    exploration_power=args.a2_exploration_power,
    exploration_const=args.a2_exploration_const
)

a1.policy_net.train()
a2.policy_net.train()

reinforce(
    env, a1, a2,
    args.num_episodes,
    enable_swap=args.enable_swap,
    enable_transform=args.enable_transform,
    px=args.px,
    pt=args.pt,
    checkpoint_rate=args.checkpoint_rate,
    validation_rate=args.validation_rate,
    experiment_name=args.experiment_name,
    weights_path=args.weights_path,
    weights_name=args.weights_name,
)
