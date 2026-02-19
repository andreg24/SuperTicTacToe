"""
Quick test script to verify MinMaxQ implementation works correctly.
"""

import torch
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from SuperTicTacToe.ultimatetictactoe import ultimatetictactoe

from agent import MinMaxQAgent, state_to_tensor
from train import ReplayBuffer, Experience 


def test_basic_functionality():
	print("Testing MinMaxQ implementation...")

	# Environment
	print("\n Creating environment...")
	env = ultimatetictactoe.env()
	env.reset()
	print("Environment created.")

	# Agents
	print("\n Creating MinMaxQ agents...")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	agent1 = MinMaxQAgent("player_1", player_id=0, device=device, mode='train')
	agent2 = MinMaxQAgent("player_2", player_id=1, device=device, mode='train')
	print(f" Agents created on {device}")
	print(f" Network parameters: {sum(p.numel() for p in agent1.q_network.parameters())}")

	# Test state conversion
	print("\n Testing state_to_tensor...")
	env.reset()
	state, _, _, _, _ = env.last()
	tensor = state_to_tensor(state, device)
	print(f" State tensor shape: {tensor.shape} (expected (1,3,9,9))")
	assert tensor.shape == (1, 3, 9, 9), f"Wrong shape: {tensor.shape}"
	print("state_to_tensor works.")

	# Test action selection
	print("\n Testing action selection...")
	output = agent1.pick_action(state)
	action = output['action']
	print(f" Picked action: {action}")
	assert 0 <= action <= 80 or action == -1, f"Invalid action: {action}"
	print("Action selection works.")

	# Test update with dummy batch
	print("\nTesting batch update...")
	batch_size = 4
	states = torch.rand(batch_size, 3, 9, 9)
	actions = torch.randint(0, 81, (batch_size,))
	rewards = torch.randn(batch_size)
	next_states = torch.rand(batch_size, 3, 9, 9)
	dones = torch.zeros(batch_size)
	next_masks = torch.ones(batch_size, 81, dtype=torch.bool)
	next_turns = torch.randint(0, 2, (batch_size,))
	batch = Experience(states, actions, rewards, next_states, dones, next_masks, next_turns)
	loss = agent1.update(batch)
	print(f" Update loss: {loss:.6f}")
	print("Batch update works.")

	# Test save/load
	print("\nTesting save/load...")
	import tempfile
	import os
	with tempfile.TemporaryDirectory() as tmpdir:
		path = os.path.join(tmpdir, "test_agent.pt")
		agent1.save(path)
		print(f"Saved to {path}")
		new_agent = MinMaxQAgent("test", player_id=0, device=device)
		new_agent.load(path)
		# Compare weights
		for p1, p2 in zip(agent1.q_network.parameters(), new_agent.q_network.parameters()):
			assert torch.allclose(p1, p2), "Parameters differ after load!"
		print("Save/load successful.")

	# Test a full episode
	print("\nTesting full episode...")
	env.reset()
	agents = {"player_1": agent1, "player_2": agent2}
	turn_count = 0
	for agent_name in env.agent_iter():
		state, reward, termination, truncation, info = env.last()
		if termination or truncation:
			action = None
		else:
			agent = agents[agent_name]
			output = agent.pick_action(state)
			action = output['action']
			turn_count += 1
		if isinstance(action, torch.Tensor):
			action = action.item()
		env.step(action)
	print(f"Episode completed in {turn_count} turns.")
	print(f"Final rewards: {env.rewards}")
	print("Episode runs without errors.")

	print("\n" + "="*60)
	print("ALL TESTS PASSED!")
	print("="*60)


if __name__ == "__main__":
	test_basic_functionality()