import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


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