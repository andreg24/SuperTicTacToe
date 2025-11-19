"""Small script to run the environment with random agents"""

from __future__ import annotations
from ultimatetictactoe import ultimatetictactoe
import numpy as np

env = ultimatetictactoe.env(render_mode="human")
env.reset(42)
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(reward)
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        mask = observation["action_mask"]
        action = env.action_space(agent).sample(mask)
    env.step(action)
