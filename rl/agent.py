from abc import ABC, abstractmethod
from typing import Optional
import random
import numpy as np

import torch
import torch.nn as nn

from rl.alphazero.mcts import MCTS
from rl.alphazero.utils import get_board_perspective
from utils.board_utils import relative_to_absolute


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
        self.final_linear = nn.Sequential(
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.Flatten(),
            nn.LazyLinear(500),
            nn.ReLU(),
            nn.LazyLinear(81),
        )

        self.softmax = nn.Softmax(1)

    def forward(self, state: torch.Tensor):
        """state should be tensor of shape (B, 4, 9, 9)"""
        action_mask = state[:, 2, :, :]
        action_mask_final = action_mask.reshape(state.size(0), -1).bool().clone()
        x = state
        x = self.first_conv_net(x)
        x = self.second_conv_net(x + action_mask)
        logits = self.final_linear(x + action_mask) # (B, 81)

        masked_logits = logits.masked_fill(~action_mask_final, float("-inf"))
        probs = self.softmax(masked_logits)

        if self.epsilon > 0:
            unif_probs = torch.zeros_like(probs)
            unif_probs[action_mask_final] = 1.0
            unif_probs = unif_probs / unif_probs.sum(dim=1, keepdim=True)
            probs = (1 - self.epsilon) * probs + self.epsilon * unif_probs
        return probs

class LocalPolicy(nn.Module):
    """Network taking in input observations and outputting masked probability distributions for each possible action"""

    def __init__(self, net=None, epsilon=0.1, dropout_p=0.1, Activation=nn.SiLU()):
        super(LocalPolicy, self).__init__()
        self.epsilon = epsilon
        self.dropout_p = dropout_p
        self.Activation = Activation

        #TODO
        self.local_conv_net = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            nn.LazyBatchNorm2d(),
            self.Activation,
            nn.LazyConv2d(8, 3, padding=1),
            nn.LazyBatchNorm2d(),
            self.Activation,
            nn.LazyConv2d(16, 3, padding=1),
            nn.LazyBatchNorm2d(),
            self.Activation,
        )
        self.global_conv_net = nn.Sequential(
            nn.LazyConv2d(32, 5, padding=2),
            nn.LazyBatchNorm2d(),
            self.Activation,
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.LazyConv2d(64, 7, padding=3),
            nn.LazyBatchNorm2d(),
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
        action_mask = state[:, 2, :, :]
        action_mask_final = action_mask.reshape(state.size(0), -1).bool().clone()

        local_feats = []
        for i in range(3):
            for j in range(3):
                local = state[:, :, 3*i:3*(i+1), 3*j, 3*(j+1)]
                feat = self.local_conv_net(local)
                local_feats.append(feat)
        x = torch.stack(local_feats, dim=1).view(1, 3, 3, 16, 3, 3)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(1, 16, 9, 9)
        x = self.global_conv_net(x + action_mask)
        logits = self.final_linear(x + action_mask) # (B, 81)

        masked_logits = logits.masked_fill(~action_mask_final, float("-inf"))
        probs = self.softmax(masked_logits)

        if self.epsilon > 0:
            unif_probs = torch.zeros_like(probs)
            unif_probs[action_mask_final] = 1.0
            unif_probs = unif_probs / unif_probs.sum(dim=1, keepdim=True)
            probs = (1 - self.epsilon) * probs + self.epsilon * unif_probs
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

    def pick_action(self, state):
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

    def pick_action(self, state):
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
        exploration_power: int = 6,
        policy_net: Optional[nn.Module] = None,
        force_mask: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        mode: str = "sample",
    ) -> None:
        super().__init__(name)
        self.policy_net = (
            policy_net
            if policy_net is not None
            else Policy(epsilon=epsilon**exploration_power)
        )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.SGD(
                self.policy_net.parameters(), lr=epsilon**learning_power
            )
        )
        self.force_mask = force_mask
        self.device = device if device is not None else torch.device("cpu")
        self.mode = mode

    def pick_action(self, state: torch.Tensor):
        """
        Obs is a (B, 3, 9, 9) tensor, where
        the first two channels are the players moves on the board
        and the last one is the action mask for the board
        """

        state_tensor = state_to_tensor(state)
            

        probs = self.policy_net(state_tensor)

        if self.force_mask:
            # TODO
            # probs[2] = 
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


class AlphaZeroAgent(BaseAgent):
	def __init__(self, name, env, model, player, n_searches=128):
		super().__init__(name)
		self.env = env
		self.model = model
		self.model.eval()
		self.mcts = MCTS(env, n_searches=n_searches)
		self.player = player

	def pick_action(self, state):
		board = self.env.board
		root, action_probs = self.mcts.run(self.model, self.player * -1, board)
		action = np.argmax(action_probs)
		self.env.reset(options={
			"board": board,
			"next_player": self.player
		})
		return dict(action=action)