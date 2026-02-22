import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
	def __init__(self, device, board_size=81, action_size=81):
		super(MLP, self).__init__()
		self.device = device
		self.size = board_size

		self.fc1 = nn.Linear(in_features=self.size, out_features=self.size * 2)
		self.fc2 = nn.Linear(in_features=self.size * 2, out_features=self.size * 2)

		self.action_head = nn.Linear(in_features=self.size * 2, out_features=action_size)
		self.value_head = nn.Linear(in_features=self.size * 2, out_features=1)

		self.to(device)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		
		action_logits = self.action_head(x)
		value_logits = self.value_head(x)

		return F.softmax(action_logits, dim=1), torch.tanh(value_logits)

	def predict(self, board):
		board = torch.FloatTensor(
			board.astype(np.float32)
		).to(self.device).view(1, self.size)
		self.eval()
		
		with torch.no_grad():
			pi, v = self.forward(board)

		return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]


class MLP2(nn.Module):
	def __init__(self, device, board_size=81, sub_board_size=9, action_size=81):
		super(MLP2, self).__init__()
		self.device = device
		self.board_size = board_size
		self.sub_board_size = sub_board_size

		self.sub = nn.Sequential(
			nn.Linear(sub_board_size, 32),
			nn.LayerNorm(32),
			nn.ReLU(),
			nn.Linear(32, 64),
			nn.LayerNorm(64),
			nn.ReLU()
		)

		self.super = nn.Sequential(
			nn.Linear(64 * 9, 512),
			nn.LayerNorm(512),
			nn.ReLU(),
			# nn.Dropout(),
			nn.Linear(512, 256),
			nn.LayerNorm(256),
			nn.ReLU(),
			# nn.Dropout(),
			nn.Linear(256, 256),
			nn.ReLU()
		)

		self.action_head = nn.Linear(in_features=256, out_features=action_size)
		self.value_head = nn.Linear(in_features=256, out_features=1)

		self.to(device)

	def forward(self, x, temperature=1.0):
		x = x.reshape((-1, self.sub_board_size, self.sub_board_size))
		x = self.sub(x)
		x = x.reshape((-1, self.sub_board_size * 64))
		# print(x.shape)
		x = self.super(x)

		action_logits = self.action_head(x)
		value_logits = self.value_head(x)

		action_logits *= temperature
		return F.softmax(action_logits, dim=-1), torch.tanh(value_logits)

	def predict(self, board, temperature=1.0):
		board = torch.FloatTensor(
			board.astype(np.float32)
		).to(self.device).view(1, self.board_size)
		self.eval()
		
		with torch.no_grad():
			pi, v = self.forward(board, temperature=temperature)

		return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]


class ResNet(nn.Module):
	def __init__(self, device, n_hidden=32, n_resnet_blocks=1, board_rows=9, board_cols=9, action_size=81):
		super(ResNet, self).__init__()
		self.start = nn.Sequential(
			nn.Conv2d(3, n_hidden, kernel_size=3, padding=1),
			nn.BatchNorm2d(n_hidden),
			nn.ReLU()
		)
		self.bb = nn.ModuleList([
			ResBlock(n_hidden) for _ in range(n_resnet_blocks)
		])
		self.policy_head = nn.Sequential(
			nn.Conv2d(n_hidden, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(32 * board_rows * board_cols, action_size),
			nn.Softmax(dim=-1)
		)
		self.value_head = nn.Sequential(
			nn.Conv2d(n_hidden, 3, kernel_size=3, padding=1),
			nn.BatchNorm2d(3),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(3 * board_rows * board_cols, 1),
			nn.Tanh()
		)
		self.board_rows, self.board_cols = board_rows, board_cols

	def forward(self, x):
		x = x.reshape((-1, self.board_rows, self.board_cols))
		x = torch.stack([
			(x == -1).to(torch.float32),
			(x == 0).to(torch.float32),
			(x == 1).to(torch.float32)
		])
		if x.shape[1] != 3:
			x = x.swapaxes(0, 1)
		x = self.start(x)
		for block in self.bb:
			x = block(x)
		return self.policy_head(x), self.value_head(x)

	def predict(self, board):
		# board = torch.FloatTensor(
		# 	board.astype(np.float32)
		# ).to(self.device).view(1, self.size)
		board = torch.stack((
			torch.tensor(board == -1, dtype=torch.float32),
			torch.tensor(board == 0, dtype=torch.float32),
			torch.tensor(board == 1, dtype=torch.float32)
		))
		self.eval()

		with torch.no_grad():
			pi, v = self.forward(board)

		return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]


class ResBlock(nn.Module):
	def __init__(self, n_hidden):
		super(ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(n_hidden)
		self.conv2 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(n_hidden)

	def forward(self, x):
		residual = x
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)) + residual)
		return x