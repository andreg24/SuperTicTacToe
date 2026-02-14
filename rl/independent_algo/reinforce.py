# TODO

from typing import Optional
import os
import datetime
import tqdm
import numpy as np

import torch

from utils.board import TRANSFORMATIONS
from .algo_utils import format_datetime


# def obs_to_tensor(state):
#     # consider adding a 4th channel for the turn number
#     board_tensor = torch.tensor(state['observation'])
#     action_mask_tensor = torch.tensor(state['action_mask'].reshape(9, 9)).unsqueeze(0)
#     return torch.cat((board_tensor, action_mask_tensor)).unsqueeze(0).to(torch.float32)


class Trajectory:

    def __init__(self, env, agent_1, agent_2, enable_log_prob=False):
        self.env = env
        self.agent_1 = agent_1  # default player_1
        self.agent_2 = agent_2  # default player_2
        self.enable_log_prob = enable_log_prob
        self.turn = 0

        self.trajectory = {
            "player_1": {
                "observations": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
            },
            "player_2": {
                "observations": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
            },
        }

    def _reset(self) -> None:
        self.env.reset()
        self.trajectory = {
            "player_1": {
                "observations": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
            },
            "player_2": {
                "observations": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
            },
        }
        self.transformations_schedule = {}
        self.turn = 0

    def _burnout(self, burnout_turn) -> None:
        """Delete the first phase of "burnout" by cutting the first self.burnout turns of each player, preserving players order"""
        if burnout_turn == 0:
            pass
        else:
            for pl in ["player_1", "player_2"]:
                for key in self.trajectory[pl].keys():
                    self.trajectory[pl][key] = self.trajectory[pl][key][
                        burnout_turn // 2 :
                    ]

    def _apply_transformations(self, transformation):
        self.env.apply_transformation(transformation)

    def compute(self, burnout_turn=0, transformation=None, transformation_turn=0, max_turn=None, reset=True):
        assert burnout_turn % 2 == 0, "burnout turn must be even number"
        assert transformation_turn % 2 == 0, "transformation turn must be even_number"
        
        if reset:
            self._reset()
        state = None
        action = None

        for agent in self.env.agent_iter():
            if max_turn is not None and self.turn == max_turn:
                break

            if self.turn == burnout_turn:
                self._burnout(burnout_turn)
            if transformation is not None and self.turn == transformation_turn:
                self._apply_transformations(transformation)

            state, reward, termination, truncation, info = (
                self.env.last()
            )  # get last step info

            # agent is done, no action to take
            if termination or truncation:
                action = None
            # pick action
            else:
                if agent == "player_1":
                    output = self.agent_1.pick_action(state)
                else:
                    output = self.agent_2.pick_action(state)

                action = output["action"]
                if "log_prob" in output.keys():
                    log_prob = output["log_prob"]

            # Record observation, action, reward
            if not (termination or truncation):
                self.trajectory[agent]["observations"].append(state["observation"])
                self.trajectory[agent]["actions"].append(action)
                if "log_prob" in output.keys():
                    self.trajectory[agent]["log_probs"].append(log_prob)
            if self.turn >= 2:
                self.trajectory[agent]["rewards"].append(reward)

            # turn action to int
            if isinstance(action, torch.Tensor):
                if action.device == torch.device("cuda"):
                    action = action.to(torch.device("cpu"))
                action = action.item()
            elif isinstance(action, np.ndarray):
                action = action.item()

            self.env.step(action)  # take the action (None if done)
            self.turn += 1

    def swap_players(self):
        self.agent_1, self.agent_2 = self.agent_2, self.agent_1


def generate_schedule(n, px, pt, max_semi_turn=15):
    """Generates transformations schedule for n trajectories"""

    enable_transformation = np.random.binomial(1, px, (n,))
    use_transformation = np.random.randint(0, 5, (n,)) * enable_transformation
    transformation_turns = (
        np.random.binomial(max_semi_turn, pt, (n,)) * enable_transformation * 2
    )
    return use_transformation, transformation_turns


# manipolate trajectory
def reinforce_update(agent, trajectory, gamma=0.99):
    rewards = trajectory["rewards"]
    log_probs = trajectory["log_probs"]
    log_probs_tensor = torch.cat(log_probs)
    G = torch.zeros(1)
    n = len(trajectory["observations"])
    returns = []
    for i in range(n - 1, -1, -1):
        # G = rewards[i] + G * gamma ** (i - n + 1)
        G = rewards[i] + G * gamma
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    # loss = -(G * log_probs_tensor).sum()
    loss = -(returns * log_probs_tensor).sum()
    agent.update(loss)


def reinforce(
    env,
    agent_1,
    agent_2,
    num_episodes,
    gamma=0.99,
    update1=True,
    update2=True,
    enable_swap=False,
    enable_transform=False,
    px=0.3,
    pt=0.5,
    max_semi_turn=15,
    checkpoint_rate: Optional[int] = None,
    experiment_name: Optional[str] = None,
) -> None:
    
    if checkpoint_rate is not None:
        if experiment_name is None:
            experiment_name = format_datetime(str(datetime.datetime.now()))
        # check folder existence
        checkpoint_folder = f"./rl/independent_algo/logs/checkpoints/{experiment_name}"
        if update1:
            os.makedirs(checkpoint_folder + "/agent_1", exist_ok=True)
        if update2:
            os.makedirs(checkpoint_folder + "/agent_2", exist_ok=True)


    TR = Trajectory(env, agent_1, agent_2, True)
    if enable_transform:
        schedule = generate_schedule(num_episodes, px, pt, max_semi_turn)

    for ep in tqdm.trange(num_episodes):

        # swap players at each epoch
        if enable_swap and ep > 0:
            TR.swap_players()

        # perform transformation
        if enable_transform and schedule[1][ep] != 0:
            TR.compute(
                schedule[1][ep], TRANSFORMATIONS[schedule[0][ep]], schedule[1][ep]
            )
        else:
            TR.compute()

        trajectory = TR.trajectory

        if not enable_swap or ep % 2 == 0:
            if update1:
                reinforce_update(agent_1, trajectory["player_1"], gamma)
            if update2:
                reinforce_update(agent_2, trajectory["player_2"], gamma)
        else:
            if update1:
                reinforce_update(agent_1, trajectory["player_2"], gamma)
            if update2:
                reinforce_update(agent_2, trajectory["player_1"], gamma)
        
        if checkpoint_rate is not None:
            if ep != 0 and ep%checkpoint_rate == 0 or ep==num_episodes-1:
                # save weights
                if update1:
                    torch.save(agent_1.policy_net.state_dict(), checkpoint_folder+f"/agent_1/model_{ep}.pt")
                if update2:
                    torch.save(agent_2.policy_net.state_dict(), checkpoint_folder+f"/agent_2/model_{ep}.pt")


def compute_games(env, agent1, agent2, n, enable_swap=True):
    """Returns stats for n games played between the two agents"""
    agent1.eval()
    agent2.eval()

    trajectory = Trajectory(env, agent1, agent2)

    results = np.zeros(3)
    rewards = np.zeros(n)
    rewards_count = {}
    game_turns = np.zeros(n)

    for i in range(n):
        if enable_swap and i > 0:
            trajectory.swap_players()
        trajectory.compute()
        t = trajectory.trajectory
        if not enable_swap or i % 2 == 0:
            reward = t["player_1"]["rewards"][-1]
        else:
            reward = t["player_2"]["rewards"][-1]

        rewards[i] = reward
        game_turns[i] = trajectory.turn

        if reward not in rewards_count.keys():
            rewards_count[reward] = 1
        else:
            rewards_count[reward] += 1

        if reward == 1:
            results[0] += 1
        elif reward == -1:
            results[1] += 1
        else:
            results[2] += 1
    # normalize rewards count
    for reward in rewards_count.keys():
        rewards_count[reward] *= 100/n

    return {
        "results": results * 100 / n,
        "rewards": rewards,
        "rewards_count": rewards_count,
        "game_turns": game_turns,
    }
