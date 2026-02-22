"""
Training utilities for MinMaxQ with fixed‑opponent, latest‑checkpoint,
buffer clearing, and terminal oversampling.
Compatible with sparse‑reward environment.
"""

import torch
import numpy as np
from collections import deque, namedtuple
import random
from typing import List, Optional
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from rl.minmaxq.agent import MinMaxQAgent, state_to_tensor

Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'next_action_mask', 'next_turn']
)


class ReplayBuffer:
	def __init__(self, capacity: int = 10000):
		self.buffer = deque(maxlen=capacity)

	def push(self, exp: Experience):
		self.buffer.append(exp)

	def sample(self, batch_size: int) -> Experience:
		batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
		states = torch.cat([e.state for e in batch])
		actions = torch.tensor([e.action for e in batch])
		rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
		next_states = torch.cat([e.next_state for e in batch])
		dones = torch.tensor([e.done for e in batch], dtype=torch.float32)
		next_masks = torch.stack([e.next_action_mask for e in batch])
		next_turns = torch.tensor([e.next_turn for e in batch])
		return Experience(states, actions, rewards, next_states, dones, next_masks, next_turns)

	def clear(self):
		self.buffer.clear()

	def __len__(self):
		return len(self.buffer)


class EpisodeMemory:
	def __init__(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.next_states = []
		self.dones = []
		self.next_masks = []
		self.next_turns = []

	def add(self, state, action, reward, next_state, done, next_mask, next_turn):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.next_states.append(next_state)
		self.dones.append(done)
		self.next_masks.append(next_mask)
		self.next_turns.append(next_turn)

	def get_experiences(self) -> List[Experience]:
		exps = []
		for i in range(len(self.states)):
			exps.append(Experience(
					self.states[i],
					self.actions[i],
					self.rewards[i],
					self.next_states[i],
					self.dones[i],
					self.next_masks[i],
					self.next_turns[i]
			))
		return exps

	def clear(self):
		for attr in ['states', 'actions', 'rewards', 'next_states', 'dones', 'next_masks', 'next_turns']:
			getattr(self, attr).clear()


class CheckpointPool:
	def __init__(self, max_size: int = 10):
		self.max_size = max_size
		self.checkpoints = []
		self.latest = None

	def add(self, path: Path):
		self.checkpoints.append(path)
		self.latest = path
		if len(self.checkpoints) > self.max_size:
			self.checkpoints.pop(0)

	def get_latest(self):
		return self.latest

	def __len__(self):
		return len(self.checkpoints)


def play_episode(env, agent1: MinMaxQAgent, agent2: MinMaxQAgent, memory1: Optional[EpisodeMemory] = None, memory2: Optional[EpisodeMemory] = None):
	"""
	Play one full episode. Terminal transitions are stored TWICE to prioritise them.
	Returns: (final_reward_agent1, final_reward_agent2)
	"""
	env.reset()
	if memory1:
			memory1.clear()
	if memory2:
			memory2.clear()

	agents = {"player_1": agent1, "player_2": agent2}
	memories = {"player_1": memory1, "player_2": memory2}

	prev_state = {"player_1": None, "player_2": None}
	prev_action = {"player_1": None, "player_2": None}

	final_rewards = {"player_1": 0.0, "player_2": 0.0}

	for agent_name in env.agent_iter():
		state, reward, termination, truncation, info = env.last()
		done = termination or truncation

		agent = agents[agent_name]
		memory = memories[agent_name]

		if reward != 0:
			final_rewards[agent_name] = reward

		if prev_state[agent_name] is not None and memory is not None:
			if not done:
				next_state = state_to_tensor(state, agent.device).cpu()
				next_mask = torch.tensor(state['action_mask'], dtype=torch.bool).cpu()
				next_turn = state['turn']
			else:
				next_state = prev_state[agent_name].cpu()
				next_mask = torch.zeros(81, dtype=torch.bool)
				next_turn = -1

			# Normal add
			memory.add(
				prev_state[agent_name].cpu(),
				prev_action[agent_name],
				reward,
				next_state,
				done,
				next_mask,
				next_turn
			)

			# Terminal oversampling
			if reward != 0:
				memory.add(
					prev_state[agent_name].cpu(),
					prev_action[agent_name],
					reward,
					next_state,
					done,
					next_mask,
					next_turn
				)

		if done:
			action = None
		else:
			output = agent.pick_action(state)
			action = output['action']
			prev_state[agent_name] = state_to_tensor(state, agent.device).cpu()
			prev_action[agent_name] = action

		if isinstance(action, torch.Tensor):
			action = action.item()
		if isinstance(action, np.ndarray):
			action = action.item()

		env.step(action)

	return final_rewards["player_1"], final_rewards["player_2"]


def train_minmaxq(env, agent1: MinMaxQAgent, agent2: MinMaxQAgent, num_episodes: int = 1000, batch_size: int = 32, update_freq: int = 4, 
	target_update_freq: int = 2000, buffer_capacity: int = 100000, enable_swap: bool = True, eval_freq: int = 2000, eval_episodes: int = 20,
	fixed_opponent: bool = False, fixed_phase_episodes: int = 5000, pool_size: int = 30, ) -> dict:
	"""
	Fixed‑opponent training with:
	- Latest checkpoint opponent (always strongest).
	- Replay buffer cleared at phase start.
	- Terminal oversampling.
	"""
	buffer1 = ReplayBuffer(buffer_capacity)
	buffer2 = ReplayBuffer(buffer_capacity)
	memory1 = EpisodeMemory()
	memory2 = EpisodeMemory()

	stats = {
		'episode_rewards_1': [],
		'episode_rewards_2': [],
		'episode_lengths': [],
		'losses_1': [],
		'losses_2': [],
		'eval_win_rates': [],
		'eval_episodes_list': [],
		'random_eval_win_rates': [],
		'random_eval_episodes': [],
	}

	if fixed_opponent:
			enable_swap = False

	checkpoint_dir = Path('rl/minmaxq/checkpoints')
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	pool1 = CheckpointPool(max_size=pool_size)
	pool2 = CheckpointPool(max_size=pool_size)

	init_path1 = checkpoint_dir / 'agent1_init.pt'
	init_path2 = checkpoint_dir / 'agent2_init.pt'
	torch.save(agent1.q_network.state_dict(), init_path1)
	torch.save(agent2.q_network.state_dict(), init_path2)
	pool1.add(init_path1)
	pool2.add(init_path2)

	agent1.mode = 'train'
	agent2.mode = 'train'

	total_steps = 0
	best_win_rate = 0.0
	best_model_path = Path('rl/minmaxq/weights/best_agent.pt')
	best_model_path.parent.mkdir(parents=True, exist_ok=True)

	for episode in range(num_episodes):
		if fixed_opponent:
				phase = episode // fixed_phase_episodes
				learner_is_agent1 = (phase % 2 == 0)
				if learner_is_agent1:
						learner, frozen = agent1, agent2
						frozen_path = pool2.get_latest()
						if frozen_path is not None:
							frozen.q_network.load_state_dict(torch.load(frozen_path, map_location=frozen.device))
						memory_learner, memory_frozen = memory1, None
						buffer_learner = buffer1
						other_buffer = buffer2
						learner.mode = 'train'
						frozen.mode = 'eval'
				else:
						learner, frozen = agent2, agent1
						frozen_path = pool1.get_latest()
						if frozen_path is not None:
							frozen.q_network.load_state_dict(torch.load(frozen_path, map_location=frozen.device))
						memory_learner, memory_frozen = memory2, None
						buffer_learner = buffer2
						other_buffer = buffer1
						learner.mode = 'train'
						frozen.mode = 'eval'

				# Clear learner's buffer at phase start
				if episode % fixed_phase_episodes == 0:
					buffer_learner.clear()
					print(f"\n🔄 Cleared learner's replay buffer for phase starting at episode {episode}")
		else:
				if enable_swap and episode % 2 == 0:
					agent1, agent2 = agent2, agent1
					buffer1, buffer2 = buffer2, buffer1
					memory1, memory2 = memory2, memory1
				agent1.mode = 'train'
				agent2.mode = 'train'
				learner = None

		# Play episode
		if fixed_opponent:
			reward1, reward2 = play_episode(
				env, agent1, agent2,
				memory1 if learner_is_agent1 else None,
				memory2 if not learner_is_agent1 else None
			)
			episode_length = (len(memory1.states) if learner_is_agent1 else len(memory2.states))
		else:
			reward1, reward2 = play_episode(env, agent1, agent2, memory1, memory2)
			episode_length = len(memory1.states) + len(memory2.states)

		stats['episode_rewards_1'].append(reward1)
		stats['episode_rewards_2'].append(reward2)
		stats['episode_lengths'].append(episode_length)

		# Store experiences
		if not fixed_opponent or learner_is_agent1:
			for exp in memory1.get_experiences():
				buffer1.push(exp)
		if not fixed_opponent or not learner_is_agent1:
			for exp in memory2.get_experiences():
				buffer2.push(exp)

		total_steps += episode_length

		# Update networks
		if not fixed_opponent or learner_is_agent1:
			if len(buffer1) >= batch_size and total_steps % update_freq == 0:
				batch = buffer1.sample(batch_size)
				loss1 = agent1.update(batch)
				stats['losses_1'].append(loss1)
		if not fixed_opponent or not learner_is_agent1:
			if len(buffer2) >= batch_size and total_steps % update_freq == 0:
				batch = buffer2.sample(batch_size)
				loss2 = agent2.update(batch)
				stats['losses_2'].append(loss2)

		# Update target networks
		if not fixed_opponent:
			if episode % target_update_freq == 0:
				agent1.update_target_network(tau=1.0)
				agent2.update_target_network(tau=1.0)
		else:
			if episode % target_update_freq == 0:
				if learner_is_agent1:
					agent1.update_target_network(tau=1.0)
				else:
					agent2.update_target_network(tau=1.0)

		# Decay epsilon
		if not fixed_opponent:
			agent1.decay_epsilon()
			agent2.decay_epsilon()
		else:
			learner.decay_epsilon()

		# Save checkpoint at phase end
		if fixed_opponent and (episode + 1) % fixed_phase_episodes == 0:
			checkpoint_path = checkpoint_dir / f'agent{1 if learner_is_agent1 else 2}_ep{episode+1}.pt'
			torch.save(learner.q_network.state_dict(), checkpoint_path)
			if learner_is_agent1:
				pool1.add(checkpoint_path)
				print(f"\n📦 Added Agent1 checkpoint (ep {episode+1}) to pool (latest is now ep {episode+1})")
			else:
				pool2.add(checkpoint_path)
				print(f"\n📦 Added Agent2 checkpoint (ep {episode+1}) to pool (latest is now ep {episode+1})")

		# Self‑play evaluation
		if episode % eval_freq == 0 and episode > 0:
			win_rate = evaluate_agents(env, agent1, agent2, num_episodes=eval_episodes)
			stats['eval_win_rates'].append(win_rate)
			stats['eval_episodes_list'].append(episode)

			print(f"\nEpisode {episode}: Win rate (agent1 vs agent2): "
						f"{win_rate['agent1_wins']:.1f}% / {win_rate['agent2_wins']:.1f}% / "
						f"{win_rate['draws']:.1f}%, Epsilon: {agent1.epsilon_current:.3f}")

			if not fixed_opponent:
				agent1.mode = 'train'
				agent2.mode = 'train'

		# Random evaluation (every 2 eval cycles)
		if episode % (eval_freq * 2) == 0 and episode > 0:
				random_result = evaluate_vs_random(env, agent1, num_episodes=20)
				stats['random_eval_win_rates'].append(random_result)
				stats['random_eval_episodes'].append(episode)

				print(f"  → vs Random: Wins: {random_result['agent_wins']:.1f}%, "
							f"Losses: {random_result['random_wins']:.1f}%, "
							f"Draws: {random_result['draws']:.1f}%")

				if random_result['agent_wins'] > best_win_rate:
					best_win_rate = random_result['agent_wins']
					torch.save(agent1.q_network.state_dict(), best_model_path)
					print(f"  🏆 New best model saved! Win rate: {best_win_rate:.1f}%")

	return stats


def evaluate_agents(env, agent1: MinMaxQAgent, agent2: MinMaxQAgent, num_episodes: int = 100) -> dict:
	original_mode1 = agent1.mode
	original_mode2 = agent2.mode
	agent1.mode = 'eval'
	agent2.mode = 'eval'

	results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
	for _ in range(num_episodes):
		r1, r2 = play_episode(env, agent1, agent2)
		if r1 > r2:
			results['agent1_wins'] += 1
		elif r2 > r1:
			results['agent2_wins'] += 1
		else:
			results['draws'] += 1

	total = sum(results.values())
	if total > 0:
		for k in results:
			results[k] = results[k] / total * 100

	agent1.mode = original_mode1
	agent2.mode = original_mode2
	return results


def evaluate_vs_random(env, agent: MinMaxQAgent, num_episodes: int = 100) -> dict:
	class RandomAgent:
		def __init__(self):
			self.name = "Random"
			self.device = torch.device('cpu')
		def pick_action(self, state, **kwargs):
			mask = state['action_mask']
			legal = np.where(mask)[0]
			if len(legal) > 0:
				action = np.random.choice(legal)
			else:
				action = -1
			return {'action': action, 'epsilon': 0.0}

	random_agent = RandomAgent()
	original_mode = agent.mode
	agent.mode = 'eval'
	results = {'agent_wins': 0, 'random_wins': 0, 'draws': 0}

	for _ in range(num_episodes):
		r1, r2 = play_episode(env, agent, random_agent)
		if r1 > r2:
			results['agent_wins'] += 1
		elif r2 > r1:
			results['random_wins'] += 1
		else:
			results['draws'] += 1

	total = sum(results.values())
	if total > 0:
		for k in results:
			results[k] = results[k] / total * 100

	agent.mode = original_mode
	return results