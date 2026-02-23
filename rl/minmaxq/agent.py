"""
MinMaxQ Agent for Ultimate Tic Tac Toe.

Single Q-network with state representation (3,9,9):
    channel 0: player 1 marks
    channel 1: player 2 marks
    channel 2: turn (1 if player1 to move, 0 if player2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from rl.agent import BaseAgent


class QNetwork(nn.Module):
	"""3‑channel input, 81 Q‑values output. Standard capacity – prevents overfitting."""
	def __init__(self, hidden_dim=128):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.bn3 = nn.BatchNorm2d(64)

		self.fc1 = nn.Linear(64 * 9 * 9, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, 81)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		return self.fc2(x)


def state_to_tensor(state: dict, device: torch.device) -> torch.Tensor:
	"""
	Convert environment state to network input.
	Returns tensor of shape (1, 3, 9, 9).
	"""
	board = torch.tensor(state['observation'], dtype=torch.float32, device=device)
	turn = torch.full((1, 9, 9), state['turn'], dtype=torch.float32, device=device)
	return torch.cat([board, turn], dim=0).unsqueeze(0)


class MinMaxQAgent(BaseAgent):
	"""
	MinMaxQ agent with double DQN and true minimax backup.
	Single network used for both players – the turn channel tells perspective.
	"""

	def __init__(self, name: str, player_id: int, learning_rate: float = 1e-3, gamma: float = 0.99, epsilon_start: float = 1.0,
		epsilon_end: float = 0.1, epsilon_decay: float = 0.9999, device: Optional[torch.device] = None, mode: str = 'train', use_double_dqn: bool = True,):
		super().__init__(name)
		# self.name = name
		self.player_id = player_id
		self.device = device if device is not None else torch.device('cpu')
		self.mode = mode
		self.gamma = gamma
		self.use_double_dqn = use_double_dqn

		self.q_network = QNetwork().to(self.device)
		self.target_network = QNetwork().to(self.device)
		self.target_network.load_state_dict(self.q_network.state_dict())
		self.target_network.eval()

		self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay
		self.epsilon = epsilon_start
		self.episodes = 0

	@property
	def epsilon_current(self):
		if self.mode == 'eval':
			return 0.0
		return max(self.epsilon_end, self.epsilon)

	def decay_epsilon(self):
		if self.epsilon > self.epsilon_end:
			self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
		self.episodes += 1

	def train(self):
		self.mode = 'train'
		self.q_network.train()

	def eval(self):
		self.mode = 'eval'
		self.q_network.eval()

	def pick_action(self, state: dict, *args, temperature: float = 1.0) -> dict:
		state_tensor = state_to_tensor(state, self.device)
		mask = torch.tensor(state['action_mask'], dtype=torch.bool, device=self.device)

		epsilon = self.epsilon_current
		if self.mode == 'train' and np.random.random() < epsilon:
			legal = torch.where(mask)[0]
			if len(legal) > 0:
				action = legal[torch.randint(len(legal), (1,))].item()
			else:
				action = -1
			return {'action': action, 'epsilon': epsilon}

		with torch.no_grad():
			q = self.q_network(state_tensor).squeeze(0)
			q[~mask] = -float('inf')
			q = torch.softmax(q, dim=-1)
			# action = q.argmax().item()
			action = np.random.choice(np.array([i for i in range(81)]), size=1, p=q.numpy())
		return {'action': action, 'epsilon': epsilon}

	def update(self, batch):
		# MinMax Policy implementation
		# Unpack batch
		states = batch.state.to(self.device)
		actions = batch.action.to(self.device).long()
		rewards = batch.reward.to(self.device).float()
		next_states = batch.next_state.to(self.device)
		dones = batch.done.to(self.device).float()
		next_masks = batch.next_action_mask.to(self.device).bool()
		next_turns = batch.next_turn.to(self.device).long()
		
		# Q-value for taken action
		q_all = self.q_network(states) 
		q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
		
		with torch.no_grad():
			if self.use_double_dqn:
				# Online network selects best action
				q_next_online = self.q_network(next_states)
				q_next_online[~next_masks] = -float('inf') # Taking away the illigal actions

				# Target network evaluates the best action (reduces overestimation bias)
				best_actions = q_next_online.argmax(dim=1)
				q_next_target = self.target_network(next_states) # Evaluate the best action
				q_next = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
			else:

				q_next = self.target_network(next_states)
				q_next[~next_masks] = -float('inf')
				q_next = q_next.max(dim=1)[0]

			current_turn = states[:, 2, 0, 0].long()
			V = torch.zeros_like(rewards)
			for i in range(len(rewards)):
				if dones[i]:
					V[i] = 0.0
				else:
					if next_turns[i] == current_turn[i]:
						V[i] = q_next[i] # It's our turn (MAX)
					else:
						V[i] = -q_next[i] # Opponent's turn (MIN)

			target = rewards + self.gamma * V * (1 - dones)
		
		# MSE loss, backprop with gradient clipping
		loss = F.mse_loss(q_sa, target)
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
		self.optimizer.step()

		return loss.item()

	def update_target_network(self, tau=1.0):
		for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

	def save(self, path: str):
		torch.save({
			'q_network': self.q_network.state_dict(),
			'target_network': self.target_network.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'epsilon': self.epsilon,
			'episodes': self.episodes,
			'player_id': self.player_id,
			'name': self.name,
		}, path)

	def load(self, path: str):
		checkpoint = torch.load(path, map_location=self.device)
		self.q_network.load_state_dict(checkpoint['q_network'])
		self.target_network.load_state_dict(checkpoint['target_network'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.epsilon = checkpoint['epsilon']
		self.episodes = checkpoint['episodes']
		self.player_id = checkpoint.get('player_id', self.player_id)
		self.name = checkpoint.get('name', self.name)