from abc import ABC, abstractmethod
from typing import Optional
import random
import numpy as np
import cloudpickle
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import torch
import torch.nn as nn

from utils.board_utils import Reflection, Rotation
from rl.alphazero.mcts import MCTS
from rl.alphazero.utils import get_board_perspective
from utils.board_utils import relative_to_absolute, split_subboards, merge_subboards, Rotation, Reflection


class Trajectory:
    """
    Docstring for Trajectory
    """

    def __init__(self, env, agent_1, agent_2, enable_log_prob=False, enable_tansform=False):
        self.env = env
        self.agent_1 = agent_1  # default player_1
        self.agent_2 = agent_2  # default player_2
        self.enable_log_prob = enable_log_prob
        self.enable_transform = enable_tansform
        self.turn = 0

        self.trajectory = {
            f"player_{i+1}": {
                "observations": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
            } 
            for i in range(2)
        }

    def _reset(self) -> None:
        self.env.reset()
        self.trajectory = {
            f"player_{i+1}": {
                "observations": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
            } 
            for i in range(2)
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

        rotation = Rotation(0)
        reflection = Reflection(0)

        for agent in self.env.agent_iter():
            k_rot = random.randint(0, 3)
            k_ref = random.randint(0, 4)
            rotation.k = k_rot
            reflection.k = k_ref

            if max_turn is not None and self.turn == max_turn:
                break

            # if self.turn == burnout_turn:
            #     self._burnout(burnout_turn)
            # if transformation is not None and self.turn == transformation_turn:
            #     self._apply_transformations(transformation)

            state, reward, termination, truncation, info = (
                self.env.last()
            )  # get last step info

            # agent is done, no action to take
            if termination or truncation:
                action = None
            # pick action
            else:
                if agent == "player_1":
                    if self.enable_transform:
                        output = self.agent_1.pick_action(state, rotation, reflection)
                    else:
                        output = self.agent_1.pick_action(state, None, None)
                else:
                    if self.enable_transform:
                        output = self.agent_2.pick_action(state, rotation, reflection)
                    else:
                        output = self.agent_2.pick_action(state, None, None)

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

def epsilon_greedy(epsilon: float, probs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:

    unif = action_mask.float()
    unif = unif / unif.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (1-epsilon)*probs + epsilon*unif


# contracting_net = nn.Sequential(
#     nn.Conv2d(4, 2, 5, padding=3),
#     nn.LazyBatchNorm2d(),
#     nn.ReLU(),
#     nn.Dropout(self.dropout_p),
#     nn.LazyConv2d(2, 5, padding=2),
#     nn.LazyBatchNorm2d(),
#     nn.ReLU(),
#     nn.Dropout(self.dropout_p),
#     nn.LazyConv2d(3, 5, padding=1),
#     nn.LazyBatchNorm2d(),
#     nn.ReLU(),
#     nn.LazyConv2d(1, 3, padding=1),
#     nn.LazyBatchNorm2d(),
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.LazyLinear(81),
# )

class SimplePolicy(nn.Module):
    """Network taking in input observations and outputting masked probability distributions for each possible action"""

    def __init__(self, net=None, epsilon=0.1, dropout_p=0.1):
        super(SimplePolicy, self).__init__()
        self.epsilon = epsilon
        self.dropout_p = dropout_p
        #TODO
        if net is None:
            self.net = nn.Sequential(
                nn.Conv2d(4, 8, 3, padding=1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.LazyConv2d(16, 3, padding=1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.LazyConv2d(32, 3, padding=1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.LazyConv2d(64, 3, padding=1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(1000),
                nn.ReLU(),
                nn.LazyLinear(81),
            )
        else:
            self.net = net

        self.softmax = nn.Softmax(1)

    def forward(self, state: torch.Tensor):
        """state should be tensor of shape (B, 4, 9, 9)"""
        action_mask = state[:, 2, :, :].reshape(state.size(0), -1).bool().clone()
        logits = self.net(state)  # (B, 81)
        masked_logits = logits.masked_fill(~action_mask, float("-inf"))
        probs = self.softmax(masked_logits)

        if self.epsilon > 0:
            unif_probs = torch.zeros_like(probs)
            unif_probs[action_mask] = 1.0
            unif_probs = unif_probs / unif_probs.sum(dim=1, keepdim=True)
            probs = (1 - self.epsilon) * probs + self.epsilon * unif_probs

        return probs


class Policy(nn.Module):
    """Network taking in input observations and outputting masked probability distributions for each possible action"""

    def __init__(self, net=None, epsilon=0.1, dropout_p=0.1):
        super(Policy, self).__init__()
        self.epsilon = epsilon
        self.dropout_p = dropout_p
        self.disable_epsilon = False
        #TODO
        self.first_conv_net = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(16, 5, padding=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        self.second_conv_net = nn.Sequential(
            nn.LazyConv2d(32, 7, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.LazyConv2d(64, 9, padding=4),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        self.main_linear = nn.Sequential(
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.Flatten(),
            nn.LazyLinear(500),
            nn.ReLU(),
            nn.LazyLinear(81),
            nn.ReLU(),
        )

        self.final_linear = nn.LazyLinear(81)

        self.softmax = nn.Softmax(1)

    def forward(self, state: torch.Tensor):
        """state should be tensor of shape (B, 4, 9, 9)"""
        action_mask = state[:, 2, :, :] #(B, 9, 9)
        action_mask_final = action_mask.reshape(state.size(0), -1).bool().clone()
        x = state
        x = self.first_conv_net(x)
        x = self.second_conv_net(x)
        x = self.main_linear(x) # (B, 81)
        logits = self.final_linear(x + action_mask.reshape(-1, 81))

        masked_logits = logits.masked_fill(~action_mask_final, float("-inf"))
        probs = self.softmax(masked_logits)


        if self.epsilon > 0 and not self.disable_epsilon:
            unif_probs = torch.zeros_like(probs)
            unif_probs[action_mask_final] = 1.0
            unif_probs = unif_probs / unif_probs.sum(dim=1, keepdim=True)
            probs = (1 - self.epsilon) * probs + self.epsilon * unif_probs
        return probs

class LocalPolicy(nn.Module):
    """Network taking in input observations and outputting masked probability distributions for each possible action"""

    def __init__(
        self,
            # net: Optional[nn.Module] = None,
            epsilon: float = 0.1, # exploration rate in epsilon-greedy setting
            dropout_p: float = 0.1,
            Activation: torch.nn.functional = nn.ReLU()
        ) -> None:
        super(LocalPolicy, self).__init__()

        self.epsilon = epsilon
        self.dropout_p = dropout_p
        self.Activation = Activation

        # net for subboards
        self.local_conv_net = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            self.Activation,
            nn.LazyConv2d(8, 3, padding=1),
            self.Activation,
            nn.LazyConv2d(16, 3, padding=1),
            self.Activation,
        )

        self.parallel_net = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            self.Activation,
            nn.LazyConv2d(8, 3, padding=1),
            self.Activation,
            nn.LazyConv2d(16, 3, padding=1),
            self.Activation,
        )

        self.global_conv_net = nn.Sequential(
            nn.LazyConv2d(32, 7, padding=3),
            self.Activation,
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.LazyConv2d(64, 9, padding=4),
            self.Activation,
        )

        self.final_linear = nn.Sequential(
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.Flatten(),
            nn.LazyLinear(512),
            self.Activation,
            nn.LazyLinear(81),
        )

        self.softmax = nn.Softmax(1)

    def forward(self, state: torch.Tensor):
        """state should be tensor of shape (B, 4, 9, 9)"""
        B, C_in, _, _ = state.shape

        mask_9x9 = state[:, 2].bool()
        mask_81 = mask_9x9.reshape(B, -1).clone()
        # subboards
        sub = split_subboards(state).view(B*9, C_in, 3, 3)
        sub_feat = self.local_conv_net(sub)
        C_out = sub_feat.size(1)
        sub_feat = sub_feat.view(B, 9, C_out, 3, 3)

        x = merge_subboards(sub_feat) # (B, C_out, 9, 9)

        par = self.parallel_net(state)

        x = torch.cat([x, par], dim=1)

        x = self.global_conv_net(x)

        logits = self.final_linear(x)

        masked_logits = logits.masked_fill(~mask_81, float("-inf"))
        probs = self.softmax(masked_logits)

        if self.epsilon > 0:
            probs = epsilon_greedy(self.epsilon, probs, mask_81)
        return probs


class BaseAgent(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def pick_action(self, state: dict) -> dict:
        """Takes in input a dictionary and outputs a dictionary with required key 'action'"""
        pass

    def eval(self):
        pass


class EnvRandomAgent:
    """Agent that picks actions uniformly at random"""

    def __init__(self, name, action_mask_enabled=True):
        self.name = name
        self.action_mask_enabled = action_mask_enabled

    def pick_action(self, env):
        if self.action_mask_enabled:
            action_mask = env.last()[0]["action_mask"]
            return env.action_space(self.name).sample(action_mask)
        else:
            return env.action_space(self.name).sample()


class RandomAgent(BaseAgent):
    """Agent that picks actions uniformly at random"""

    def __init__(self, name, action_mask_enabled=True):
        super().__init__(name)
        self.action_mask_enabled = action_mask_enabled

    def pick_action(self, state, *args, **kwargs):
        if not self.action_mask_enabled:
            action = np.random.randint(0, 81)
        else:
            mask = state["action_mask"]
            valid_actions = np.where(mask == 1)[0]
            action = np.random.choice(valid_actions)
        return {"action": action}


class ManualAgent(BaseAgent):
    """Agent that picks actions uniformly at random"""

    def __init__(self, name):
        super().__init__(name)

    def pick_action(self, state, *args, **kwargs):
        action = input("insert position: ")
        super_pos, sub_pos = action.split(" ")
        super_pos = int(super_pos)
        sub_pos = int(sub_pos)
        action = relative_to_absolute(super_pos, sub_pos)
        return {"action": action}


def state_to_tensor(
    state: dict[np.ndarray],
    turn_enabled=True, # if adding turn info to tensor
    dtype=torch.float32,
    device=torch.device("cpu"),
):
    """Takes the state/observation dict and returns a tensor"""

    # consider adding a 4th channel for the turn number
    board_tensor = torch.tensor(state["observation"])
    action_mask_tensor = torch.tensor(state["action_mask"].reshape(9, 9)).unsqueeze(0)
    if turn_enabled:
        turn_tensor = torch.ones(1, 9, 9) * state["turn"]

    if turn_enabled:
        state_tensor = torch.cat((board_tensor, action_mask_tensor, turn_tensor))
    else:
        state_tensor = torch.cat((board_tensor, action_mask_tensor))

    state_tensor = state_tensor.unsqueeze(0)
    state_tensor = state_tensor.to(dtype=dtype, device=device)

    return state_tensor


class NeuralAgent(BaseAgent):
    """Agent with a neural network evaluating policy(action/state)"""

    def __init__(
        self,
        name: str,
        epsilon: float = 0.1,
        learning_power: int = 2,
        learning_const: float = 1.0,
        exploration_power: int = 6,
        exploration_const: float = 1.0,
        policy_net: Optional[nn.Module] = None,
        force_mask: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        mode: str = "sample",
    ) -> None:
        self.epsilon = epsilon
        self.learning_power = learning_power
        self.learning_const = learning_const
        self.exploration_power = exploration_power
        self.exploration_const = exploration_const
        super().__init__(name)
        self.policy_net = (
            policy_net
            if policy_net is not None
            else Policy(epsilon=exploration_const*epsilon**exploration_power)
        )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.SGD(
                self.policy_net.parameters(), lr=learning_const*epsilon**learning_power
            )
        )
        self.force_mask = force_mask
        self.device = device if device is not None else torch.device("cpu")
        self.mode = mode

    def pick_action(self, state: torch.Tensor, rotation: Optional[Rotation] = None, reflection: Optional[Reflection] = None):
        """
        Obs is a (B, 3, 9, 9) tensor, where
        the first two channels are the players moves on the board
        and the last one is the action mask for the board
        """

        state_tensor = state_to_tensor(state)

        if rotation is not None and reflection is not None:
            state_tensor_trans = reflection.transform_batch(rotation.transform_batch(state_tensor))
        elif rotation is not None:
            state_tensor_trans = rotation.transform_batch(state_tensor)
        elif reflection is not None:
            state_tensor_trans = reflection.transform_batch(state_tensor)
        else:
            state_tensor_trans = state_tensor


        probs_trans = self.policy_net(state_tensor_trans)

        if rotation is not None and reflection is not None:
            probs = rotation.inverse_transform_batch(reflection.inverse_transform_batch(probs_trans.reshape(-1, 9, 9))).reshape(-1, 81)
        elif rotation is not None:
            probs = rotation.inverse_transform_batch(probs_trans.reshape(-1, 9, 9)).reshape(-1, 81)
        elif reflection is not None:
            probs = reflection.inverse_transform_batch(probs_trans.reshape(-1, 9, 9)).reshape(-1, 81)
        else:
            probs = probs_trans

        mask_trans = state_tensor_trans[:, 2].bool().reshape(probs_trans.size(0), -1)
        assert (probs_trans[~mask_trans] == 0).all(), "policy outputs illegal mass in transformed frame"

        mask_orig = state_tensor[:, 2].bool().reshape(probs.size(0), -1)
        assert (probs[~mask_orig] == 0).all(), "inverse-transform misaligned with original mask"

        


        if self.force_mask:
            # TODO
            pass

        dist = torch.distributions.Categorical(probs)
        if self.mode == 'sample':
            action = dist.sample()
        elif self.mode == 'argmax':
            action = torch.argmax(probs)
        log_prob = dist.log_prob(action)
        return {"action": action, "log_prob": log_prob, "probs": probs}

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def eval(self):
        self.policy_net.eval()
    
    def train(self):
        self.policy_net.train()
    
    def load(self, weights_path):
        self.policy_net.load_state_dict(torch.load(weights_path))


class AlphaZeroAgent(BaseAgent):
	def __init__(self, name, env, model, player, n_searches=128):
		super().__init__(name)
		self.env = env
		self.model = model
		self.model.eval()
		self.mcts = MCTS(env, n_searches=n_searches)
		self.player = player

	def pick_action(self, state, *args):
		board = self.env.board
		root, action_probs = self.mcts.run(self.model, self.player * -1, board)
		action = np.argmax(action_probs)
		self.env.reset(options={
			"board": board,
			"next_player": self.player
		})
		return dict(action=action)


def compute_games(env, agent1: NeuralAgent, agent2: NeuralAgent, n, enable_swap=True, verbose=True):
    """Returns stats for n games played between the two agents"""
    agent1.eval()
    agent2.eval()

    if isinstance(agent1, NeuralAgent):
        agent1.policy_net.disable_epsilon = True
    if isinstance(agent2, NeuralAgent):
        agent2.policy_net.disable_epsilon = True

    trajectory = Trajectory(env, agent1, agent2)

    results = np.zeros(3)
    rewards = np.zeros(n)
    rewards_count = {}
    game_turns = np.zeros(n)

    for i in range(n):
        if verbose:
            print(f"Game {i+1}/{n}")
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
    
    if isinstance(agent1, NeuralAgent):
        agent1.policy_net.disable_epsilon = False
    if isinstance(agent2, NeuralAgent):
        agent2.policy_net.disable_epsilon = False

    return {
        "results": results * 100 / n,
        "rewards": rewards,
        "rewards_count": rewards_count,
        "game_turns": game_turns,
    }

def _compute_game_func(env_fn, agent1_fn, agent2_fn, enable_swap=True, verbose=True):
	env = cloudpickle.loads(env_fn)()
	agent1 = cloudpickle.loads(agent1_fn)(env)
	agent2 = cloudpickle.loads(agent2_fn)(env)
	return compute_games(env, agent1, agent2, 1, enable_swap=enable_swap, verbose=verbose)

def async_compute_games(env_fn, agent1_fn, agent2_fn, n_games, n_processes, enable_swap=True, verbose=True):
	with mp.Pool(processes=n_processes) as pool:
		results = pool.starmap(_compute_game_func, [(
			cloudpickle.dumps(env_fn),
			cloudpickle.dumps(agent1_fn),
			cloudpickle.dumps(agent2_fn),
			enable_swap,
			False
		) for _ in range(n_games)])
	stats = {
		"results": [0.0, 0.0, 0.0],
		"rewards": [],
		"rewards_count": [],
		"game_turns": []
	}
	for result in results:
		stats["results"] = [(o1 * len(results) + o2) / len(results)  for o1, o2 in zip(stats["results"], result["results"])]
		stats["rewards"].extend(result["rewards"])
		stats["game_turns"].extend(result["game_turns"])

	rewards, counts = np.unique(stats["rewards"], return_counts=True)
	stats["rewards_count"] = {
		r: c for r, c in zip(rewards, counts)
	}
	return stats
