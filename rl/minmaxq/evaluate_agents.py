"""
Evaluation script to compare different RL approaches.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from SuperTicTacToe.ultimatetictactoe import ultimatetictactoe

from agent import MinMaxQAgent
from train import train_minmaxq, evaluate_vs_random


def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	env = ultimatetictactoe.env()

	agents = []

	# Load MinMaxQ agents
	if args.minmaxq_path:
		print(f"Loading MinMaxQ agent from {args.minmaxq_path}...")
		agent = MinMaxQAgent("MinMaxQ", player_id=0, device=device, mode='eval')
		agent.load(args.minmaxq_path)
		agents.append(agent)

	if args.minmaxq_path2:
		print(f"Loading second MinMaxQ agent from {args.minmaxq_path2}...")
		agent2 = MinMaxQAgent("MinMaxQ2", player_id=0, device=device, mode='eval')
		agent2.load(args.minmaxq_path2)
		agents.append(agent2)

	# Random baseline
	if args.include_random:
		print("Adding random agent...")
		class RandomAgent:
			name = "Random"
			device = torch.device('cpu')
			def pick_action(self, state, **kwargs):
				mask = state['action_mask']
				legal = np.where(mask)[0]
				action = np.random.choice(legal) if len(legal) > 0 else -1
				return {'action': action}
		agents.append(RandomAgent())

	if len(agents) < 2:
			print("Error: Need at least 2 agents to evaluate.")
			return

	# Head-to-head evaluation
	if len(agents) == 2:
		print(f"\nEvaluating {agents[0].name} vs {agents[1].name}...")
		results = evaluate_agents(env, agents[0], agents[1], num_episodes=args.num_games)
		print(f"\nResults:")
		print(f"  {agents[0].name} wins: {results['agent1_wins']:.1f}%")
		print(f"  {agents[1].name} wins: {results['agent2_wins']:.1f}%")
		print(f"  Draws: {results['draws']:.1f}%")
	else:
		print("\nRunning round-robin tournament...")
		n = len(agents)
		matrix = pd.DataFrame(index=[a.name for a in agents], columns=[a.name for a in agents])
		for i in range(n):
			for j in range(n):
				if i == j:
					continue
				res = evaluate_agents(env, agents[i], agents[j], num_episodes=args.num_games)
				matrix.iloc[i, j] = f"{res['agent1_wins']:.0f}%"
		print("\nWin percentages (row player vs column player):")
		print(matrix.fillna('-'))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--minmaxq_path', type=str, help='mimmaxq/weights_diverse_opponent/weights/agent1.pt')
	parser.add_argument('--minmaxq_path2', type=str, help='mimmaxq/weights_diverse_opponent/weights/agent2.pt')
	parser.add_argument('--include_random', action='store_true', help='Include random baseline')
	parser.add_argument('--num_games', type=int, default=100, help='Number of games per matchup')
	args = parser.parse_args()
	main(args)