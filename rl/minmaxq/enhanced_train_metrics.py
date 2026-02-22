"""
Enhanced metrics tracking for MinMaxQ training.
Add this to your train.py
"""

import torch
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from collections import defaultdict

# Aggiungi path solo se necessario
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class MetricsTracker:
	"""Comprehensive metrics tracking for RL training."""
	
	def __init__(self):
		self.metrics = defaultdict(list)
			
	def record(self, key, value):
		"""Record a single metric value."""
		self.metrics[key].append(value)
	
	def get_summary(self):
		"""Get summary statistics for all metrics."""
		summary = {}
		for key, values in self.metrics.items():
			if len(values) > 0:
				summary[key] = {
					'mean': np.mean(values),
					'std': np.std(values),
					'min': np.min(values),
					'max': np.max(values),
					'last': values[-1]
				}
		return summary


def compute_episode_metrics(memory1, memory2, final_rewards):
	"""
	Compute metrics from a single episode.
	
	Returns:
			dict with episode-level metrics
	"""
	metrics = {}
	
	# Episode length
	total_turns = len(memory1.states) + len(memory2.states)
	metrics['episode_length'] = total_turns
	
	# Rewards
	metrics['reward_agent1'] = final_rewards['player_1']
	metrics['reward_agent2'] = final_rewards['player_2']
	
	# Game outcome
	if final_rewards['player_1'] > 0:
		metrics['winner'] = 1
		metrics['turns_to_win'] = total_turns
	elif final_rewards['player_2'] > 0:
		metrics['winner'] = 2
		metrics['turns_to_win'] = total_turns
	else:
		metrics['winner'] = 0  # Draw
		metrics['turns_to_win'] = None
	
	return metrics


def compute_q_value_statistics(agent, states_batch):
	"""
	Compute Q-value statistics from a batch of states.
	
	Args:
			agent: MinMaxQAgent
			states_batch: Batch of states (B, 3, 9, 9)
	
	Returns:
			dict with Q-value statistics
	"""
	with torch.no_grad():
		# Sposta il batch sul device dell'agent
		states_batch = states_batch.to(agent.device)  
		q_values = agent.q_network(states_batch)  # (B, 81)
		
		stats = {
			'q_mean': q_values.mean().item(),
			'q_std': q_values.std().item(),
			'q_min': q_values.min().item(),
			'q_max': q_values.max().item(),
		}
	
	return stats


def compute_action_distribution(actions, action_mask):
	"""
	Analyze action distribution.
	
	Args:
			actions: Tensor of actions taken (B,)
			action_mask: Valid actions mask (B, 81)
	
	Returns:
			dict with action statistics
	"""
	stats = {}
	
	# Center position
	stats['center_plays'] = (actions == 40).float().mean().item()
	
	# Corner positions (0, 2, 6, 8, 18, 20, 24, 26, ...)
	corners = [0, 2, 6, 8, 18, 20, 24, 26, 36, 38, 42, 44, 54, 56, 60, 62, 72, 74, 78, 80]
	is_corner = torch.zeros_like(actions, dtype=torch.bool)
	for corner in corners:
		is_corner |= (actions == corner)
	stats['corner_plays'] = is_corner.float().mean().item()
	
	# Edge positions
	edges = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25, ...]  
	# Simplified: not corner, not center
	stats['edge_plays'] = 1.0 - stats['center_plays'] - stats['corner_plays']
	
	return stats


def compute_gradient_norm(agent):
	"""
	Compute the norm of gradients (measure of training stability).
	
	Args:
			agent: MinMaxQAgent with gradients computed
	
	Returns:
			float: gradient norm
	"""
	total_norm = 0.0
	for p in agent.q_network.parameters():
		if p.grad is not None:
			param_norm = p.grad.data.norm(2)
			total_norm += param_norm.item() ** 2
	total_norm = total_norm ** 0.5
	return total_norm


def evaluate_detailed(env, agent, num_episodes=100):
	"""
	Detailed evaluation with breakdown by outcome.
	
	Returns:
			dict with detailed evaluation metrics
	"""
	results = {'wins': 0, 'losses': 0, 'draws': 0, 'turns_when_win': [], 'turns_when_loss': [], 'turns_when_draw': [], 'first_moves': [],}
	
	class RandomAgent:
		def __init__(self):
			self.name = "Random"
			self.device = torch.device('cpu')
		def pick_action(self, state, **kwargs):
			mask = state['action_mask']
			legal = np.where(mask)[0]
			return {'action': np.random.choice(legal) if len(legal) > 0 else -1}
	
	random_agent = RandomAgent()
	original_mode = agent.mode
	agent.mode = 'eval'
	
	for _ in range(num_episodes):
		env.reset()
		turn_count = 0
		first_move = None
		
		agents_dict = {"player_1": agent, "player_2": random_agent}
		
		for agent_name in env.agent_iter():
			state, reward, termination, truncation, info = env.last()
			
			if termination or truncation:
				# Record outcome
				if env.rewards.get('player_1', 0) > 0:
					results['wins'] += 1
					results['turns_when_win'].append(turn_count)
				elif env.rewards.get('player_2', 0) > 0:
					results['losses'] += 1
					results['turns_when_loss'].append(turn_count)
				else:
					results['draws'] += 1
					results['turns_when_draw'].append(turn_count)
				break
			
			current_agent = agents_dict[agent_name]
			output = current_agent.pick_action(state)
			action = output['action']
			
			# Record first move
			if turn_count == 0 and agent_name == 'player_1':
				first_move = action
				results['first_moves'].append(action)
			
			if isinstance(action, torch.Tensor):
				action = action.item()
			
			env.step(action)
			turn_count += 1
	
	# Compute statistics
	total = results['wins'] + results['losses'] + results['draws']
	metrics = {
		'win_rate': results['wins'] / total * 100,
		'loss_rate': results['losses'] / total * 100,
		'draw_rate': results['draws'] / total * 100,
		'avg_turns_to_win': np.mean(results['turns_when_win']) if results['turns_when_win'] else None,
		'avg_turns_to_loss': np.mean(results['turns_when_loss']) if results['turns_when_loss'] else None,
		'avg_turns_to_draw': np.mean(results['turns_when_draw']) if results['turns_when_draw'] else None,
		'first_move_center_rate': (np.array(results['first_moves']) == 40).mean() if results['first_moves'] else 0,
	}
	
	agent.mode = original_mode
	return metrics

# Train.py modification
def enhanced_train_minmaxq(
	env,
	agent1,
	agent2,
	num_episodes=1000,
	batch_size=32,
	update_freq=4,
	target_update_freq=2000,
	buffer_capacity=100000,
	enable_swap=False,
	eval_freq=500,
	eval_episodes=20,
):
	from train import ReplayBuffer, EpisodeMemory, play_episode
	
	buffer1 = ReplayBuffer(buffer_capacity)
	buffer2 = ReplayBuffer(buffer_capacity)
	memory1 = EpisodeMemory()
	memory2 = EpisodeMemory()
	
	# Enhanced stats tracking
	tracker = MetricsTracker()
	
	stats = {
		'episode_rewards_1': [],
		'episode_rewards_2': [],
		'episode_lengths': [],
		'losses_1': [],
		'losses_2': [],
		'epsilons': [], 
		'q_value_stats': [],  
		'gradient_norms': [],  
		'eval_win_rates': [],
		'eval_episodes_list': [],
		'eval_detailed': [],  
	}
	
	total_steps = 0
	
	for episode in range(num_episodes):
		# Role swapping
		if enable_swap and episode % 2 == 0 and episode > 0:
			agent1, agent2 = agent2, agent1
			buffer1, buffer2 = buffer2, buffer1
			memory1, memory2 = memory2, memory1
		
		# Play episode
		reward1, reward2 = play_episode(env, agent1, agent2, memory1, memory2)
		
		# Compute episode metrics
		ep_metrics = compute_episode_metrics(memory1, memory2, {
			'player_1': reward1,
			'player_2': reward2
		})
		
		stats['episode_rewards_1'].append(reward1)
		stats['episode_rewards_2'].append(reward2)
		stats['episode_lengths'].append(ep_metrics['episode_length'])
		stats['epsilons'].append(agent1.epsilon_current)
		
		# Track winner
		tracker.record('winner', ep_metrics['winner'])
		if ep_metrics['turns_to_win'] is not None:
			tracker.record('turns_to_win', ep_metrics['turns_to_win'])
		
		# Store experiences
		for exp in memory1.get_experiences():
			buffer1.push(exp)
		for exp in memory2.get_experiences():
			buffer2.push(exp)
		
		total_steps += ep_metrics['episode_length']
		
		# Update networks
		if len(buffer1) >= batch_size and total_steps % update_freq == 0:
			batch = buffer1.sample(batch_size)
			loss1 = agent1.update(batch)
			stats['losses_1'].append(loss1)
			
			# Compute gradient norm
			grad_norm = compute_gradient_norm(agent1)
			stats['gradient_norms'].append(grad_norm)
			
			# Q-value statistics (every 10 updates)
			if len(stats['losses_1']) % 10 == 0:
				q_stats = compute_q_value_statistics(agent1, batch.state)
				stats['q_value_stats'].append(q_stats)
		
		if len(buffer2) >= batch_size and total_steps % update_freq == 0:
			batch = buffer2.sample(batch_size)
			loss2 = agent2.update(batch)
			stats['losses_2'].append(loss2)
		
		# Update target networks
		if episode % target_update_freq == 0 and episode > 0:
			agent1.update_target_network()
			agent2.update_target_network()
		
		# Decay epsilon
		agent1.decay_epsilon()
		agent2.decay_epsilon()
		
		# Evaluation
		if episode % eval_freq == 0 and episode > 0:
			# Detailed evaluation
			detailed_eval = evaluate_detailed(env, agent1, num_episodes=eval_episodes)
			stats['eval_detailed'].append(detailed_eval)
			stats['eval_episodes_list'].append(episode)
			
			print(f"\n{'='*60}")
			print(f"Episode {episode} - Detailed Evaluation:")
			print(f"  Win Rate: {detailed_eval['win_rate']:.1f}%")
			print(f"  Loss Rate: {detailed_eval['loss_rate']:.1f}%")
			print(f"  Draw Rate: {detailed_eval['draw_rate']:.1f}%")
			if detailed_eval['avg_turns_to_win']:
				print(f"  Avg Turns to Win: {detailed_eval['avg_turns_to_win']:.1f}")
			if detailed_eval['avg_turns_to_loss']:
				print(f"  Avg Turns to Loss: {detailed_eval['avg_turns_to_loss']:.1f}")
			print(f"  First Move Center: {detailed_eval['first_move_center_rate']*100:.1f}%")
			print(f"  Epsilon: {agent1.epsilon_current:.3f}")
			if stats['q_value_stats']:
				last_q = stats['q_value_stats'][-1]
				print(f"  Q-values: mean={last_q['q_mean']:.3f}, std={last_q['q_std']:.3f}")
			print(f"{'='*60}")
	
	return stats, tracker

def plot_enhanced_stats(stats, save_path=None):
	"""
	Create comprehensive visualization of training.
	"""	
	fig = plt.figure(figsize=(20, 12))
	gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
	
	# Episode Rewards
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(stats['episode_rewards_1'], alpha=0.3, label='Agent 1')
	ax1.plot(stats['episode_rewards_2'], alpha=0.3, label='Agent 2')
	window = 50
	if len(stats['episode_rewards_1']) > window:
		avg1 = np.convolve(stats['episode_rewards_1'], np.ones(window)/window, mode='valid')
		avg2 = np.convolve(stats['episode_rewards_2'], np.ones(window)/window, mode='valid')
		ax1.plot(range(window-1, len(stats['episode_rewards_1'])), avg1, 'b-', linewidth=2)
		ax1.plot(range(window-1, len(stats['episode_rewards_2'])), avg2, 'r-', linewidth=2)
	ax1.set_title('Episode Rewards')
	ax1.set_xlabel('Episode')
	ax1.set_ylabel('Reward')
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	
	# Episode Lengths
	ax2 = fig.add_subplot(gs[0, 1])
	ax2.plot(stats['episode_lengths'], alpha=0.4)
	if len(stats['episode_lengths']) > window:
		avg_len = np.convolve(stats['episode_lengths'], np.ones(window)/window, mode='valid')
		ax2.plot(range(window-1, len(stats['episode_lengths'])), avg_len, 'r-', linewidth=2)
	ax2.set_title('Episode Lengths')
	ax2.set_xlabel('Episode')
	ax2.set_ylabel('Turns')
	ax2.grid(True, alpha=0.3)
	
	# Epsilon Decay
	ax3 = fig.add_subplot(gs[0, 2])
	ax3.plot(stats['epsilons'], 'g-', linewidth=2)
	ax3.set_title('Epsilon Decay')
	ax3.set_xlabel('Episode')
	ax3.set_ylabel('Epsilon')
	ax3.grid(True, alpha=0.3)
	
	# Training Loss
	ax4 = fig.add_subplot(gs[1, 0])
	if stats['losses_1']:
		ax4.plot(stats['losses_1'], alpha=0.5, label='Agent 1')
		ax4.plot(stats['losses_2'], alpha=0.5, label='Agent 2')
		ax4.set_yscale('log')
		ax4.set_title('Training Loss (log scale)')
		ax4.set_xlabel('Update Step')
		ax4.set_ylabel('Loss')
		ax4.legend()
		ax4.grid(True, alpha=0.3)
	
	# Q-Value Statistics
	ax5 = fig.add_subplot(gs[1, 1])
	if stats['q_value_stats']:
		q_means = [q['q_mean'] for q in stats['q_value_stats']]
		q_stds = [q['q_std'] for q in stats['q_value_stats']]
		ax5.plot(q_means, label='Mean Q-value', linewidth=2)
		ax5.fill_between(range(len(q_means)), np.array(q_means) - np.array(q_stds), np.array(q_means) + np.array(q_stds), alpha=0.3)
		ax5.set_title('Q-Value Statistics')
		ax5.set_xlabel('Update Step (×10)')
		ax5.set_ylabel('Q-Value')
		ax5.legend()
		ax5.grid(True, alpha=0.3)
	
	# Gradient Norms
	ax6 = fig.add_subplot(gs[1, 2])
	if stats['gradient_norms']:
		ax6.plot(stats['gradient_norms'], alpha=0.5)
		if len(stats['gradient_norms']) > 50:
			avg_grad = np.convolve(stats['gradient_norms'], np.ones(50)/50, mode='valid')
			ax6.plot(range(49, len(stats['gradient_norms'])), avg_grad, 'r-', linewidth=2)
		ax6.set_title('Gradient Norms')
		ax6.set_xlabel('Update Step')
		ax6.set_ylabel('Gradient Norm')
		ax6.grid(True, alpha=0.3)
	
	# Win Rates
	ax7 = fig.add_subplot(gs[2, 0])
	if stats['eval_detailed']:
		episodes = stats['eval_episodes_list']
		win_rates = [e['win_rate'] for e in stats['eval_detailed']]
		loss_rates = [e['loss_rate'] for e in stats['eval_detailed']]
		draw_rates = [e['draw_rate'] for e in stats['eval_detailed']]
		
		ax7.plot(episodes, win_rates, 'g-o', label='Wins', linewidth=2)
		ax7.plot(episodes, loss_rates, 'r-o', label='Losses', linewidth=2)
		ax7.plot(episodes, draw_rates, 'b-o', label='Draws', linewidth=2)
		ax7.set_title('Evaluation Results vs Random')
		ax7.set_xlabel('Episode')
		ax7.set_ylabel('Percentage')
		ax7.legend()
		ax7.grid(True, alpha=0.3)
	
	# Average Turns by Outcome
	ax8 = fig.add_subplot(gs[2, 1])
	if stats['eval_detailed']:
		episodes = stats['eval_episodes_list']
		turns_win = [e['avg_turns_to_win'] for e in stats['eval_detailed'] if e['avg_turns_to_win']]
		turns_loss = [e['avg_turns_to_loss'] for e in stats['eval_detailed'] if e['avg_turns_to_loss']]
		
		if turns_win:
			ax8.plot(episodes[:len(turns_win)], turns_win, 'g-o', label='Turns to Win', linewidth=2)
		if turns_loss:
			ax8.plot(episodes[:len(turns_loss)], turns_loss, 'r-o', label='Turns to Loss', linewidth=2)
		ax8.set_title('Average Game Length by Outcome')
		ax8.set_xlabel('Episode')
		ax8.set_ylabel('Turns')
		ax8.legend()
		ax8.grid(True, alpha=0.3)
	
	# Strategic Metrics
	ax9 = fig.add_subplot(gs[2, 2])
	if stats['eval_detailed']:
		episodes = stats['eval_episodes_list']
		center_rates = [e['first_move_center_rate'] * 100 for e in stats['eval_detailed']]
		ax9.plot(episodes, center_rates, 'purple', marker='o', linewidth=2)
		ax9.set_title('First Move: Center Position Rate')
		ax9.set_xlabel('Episode')
		ax9.set_ylabel('Percentage')
		ax9.grid(True, alpha=0.3)
		ax9.axhline(y=1.23, color='r', linestyle='--', label='Random (1/81)')
		ax9.legend()
	
	plt.suptitle('MinMaxQ Training - Comprehensive Metrics', fontsize=16, fontweight='bold')
	
	if save_path:
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Enhanced plot saved to {save_path}")
	
	# plt.show()
	plt.close()


if __name__ == "__main__":
	print("Metrics enhancement module loaded successfully!")