import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse
import random
import datetime
import sys
import numpy as np
import time
import torch
import torch.optim as optim
import cloudpickle
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from ultimatetictactoe import ultimatetictactoe
from rl.alphazero.model import MLP, ResNet
from rl.alphazero.mcts import MCTS
from rl.independent_algo.reinforce import compute_games


def episode(env: ultimatetictactoe.env, model, n_searches):
	samples = []
	current_player = 1
	board = None
	mcts = MCTS(env, n_searches=n_searches)

	while True:
		state = get_board_perspective(env, current_player)
		root, action_probs = mcts.run(model, current_player, board)
		samples.append((state, current_player, action_probs))

		env.reset(options={
			"board": board,
			"next_player": current_player
		})

		action, node, _ = root.select()
		env.step(action)
		observation, reward, termination, truncation, info = env.last()
		reward = -reward

		if reward != 0 or termination:
			# print(f"Reward! Player {current_player} has won!")
			return [(
				hist_state,
				hist_action_probs,
				reward * ((-1) ** (hist_player != current_player))
			) for hist_state, hist_player, hist_action_probs in samples]
		
		current_player *= -1
		board = env.board
		# print(f"No reward, doing another search starting from root -- {action} --> {node}")

def episode_async(env_fn, model, n_searches):
	env = cloudpickle.loads(env_fn)()
	model = cloudpickle.loads(model)
	return episode(env, model, n_searches)

def _train(env: ultimatetictactoe.env, model, n_iters, n_episodes, n_epochs, n_searches, batch_size):
	for i in range(1, n_iters + 1):
		print(f"Iteration {i}/{n_iters}")

		samples = []
		for j in range(1, n_episodes + 1):
			print(f"Episode {j}/{n_episodes}")
			samples.extend(episode(env, model, n_searches))
		
		random.shuffle(samples)
		print("All episodes executed. Training...")
		train_model(model, samples, n_epochs, batch_size)
		print()

def _train_async(env_fn: callable, model, n_iters, n_episodes, n_epochs, n_searches, batch_size, n_processes=1, model_out=None):
	stats = []
	best = float("inf"), float("inf")
	for i in range(1, n_iters + 1):
		print(f"Iteration {i}/{n_iters}")

		with mp.Pool(processes=n_processes) as pool:
			results = pool.starmap(episode_async, [
				(
					cloudpickle.dumps(env_fn),
					cloudpickle.dumps(model),
					n_searches
				) for _ in range(n_episodes)
			])
		
		samples = []
		for ep_samples in results:
			samples.extend(ep_samples)
		random.shuffle(samples)
		# print("All episodes executed. Training...")
		loss_pi, loss_v = train_model(model, samples, n_epochs, batch_size)
		if loss_pi < best[0] and loss_v < best[1]:
			best = (loss_pi, loss_v)
			if model_out:
				torch.save(model.state_dict(), model_out)
		stats.append((loss_pi, loss_v))
		# print()
	return stats

# def _eval(env: ultimatetictactoe.env, model):
# 	model.eval()

# 	mcts = MCTS(env, n_searches=64)
# 	current_player = 1
# 	board = None

# 	while True:
# 		if current_player == 1:
# 			action = env.action_space(env.agent_selection).sample(
# 				env.action_mask(env.agent_selection)
# 			)
# 			# print(f"Player {current_player} (random) playing action {action}")
# 		else:
# 			# state = get_board_perspective(env, current_player)
# 			_, action_probs = mcts.run(model, current_player, board)
# 			action = np.argmax(action_probs)
# 			# print(f"Player {current_player} (model) playing action {action}")

# 		env.reset(options={
# 			"board": board,
# 			"next_player": current_player
# 		})
# 		env.step(action)
# 		observation, reward, termination, truncation, info = env.last()
# 		reward = -reward

# 		if termination:
# 			if reward > 0:
# 				print(f"Player {current_player} has won!")
# 			elif reward < 0:
# 				print(f"Player {current_player} has lost!")
# 			else:
# 				print("It's a tie!")
# 			return reward
		
# 		current_player *= -1
# 		board = env.board
def _eval(env: ultimatetictactoe.env, model, n_matches, n_searches):
	from rl.agent import AlphaZeroAgent, RandomAgent
	agent1 = AlphaZeroAgent("player_1", env, model, 1, n_searches=n_searches)
	agent2 = RandomAgent("player_2", action_mask_enabled=True)
	stats = compute_games(env, agent1, agent2, n_matches, enable_swap=False)
	print(stats)

def train_model(model, samples, n_epochs=1, batch_size=32):
	optimizer = optim.Adam(model.parameters(), lr=5e-4)
	losses_pi = []
	losses_v = []

	for epoch in range(1, n_epochs + 1):
		print(f"Epoch {epoch}/{n_epochs}")
		model.train()

		batch_idx = 0
		while batch_idx < int(len(samples) / batch_size):
			sample_idcs = np.random.randint(len(samples), size=batch_size)
			states, pis, vs = list(zip(*[samples[i] for i in sample_idcs]))
			states = torch.FloatTensor(np.array(states).astype(np.float64))
			t_pi = torch.FloatTensor(np.array(pis))
			t_v = torch.FloatTensor(np.array(vs).astype(np.float64))

			states = states.contiguous()
			t_pi = t_pi.contiguous()
			t_v = t_v.contiguous()

			p_pi, p_v = model(states)
			loss_pi = -(t_pi * torch.log(p_pi)).sum(dim=1).mean()
			loss_v = torch.sum((t_v - p_v.view(-1)) ** 2) / t_v.size()[0]
			loss_total = loss_pi + loss_v
			# print(f"p_pi = {p_pi} | p_v = {p_v} | loss_pi = {loss_pi} | loss_v = {loss_v}")
			losses_pi.append(loss_pi.detach().numpy())
			losses_v.append(loss_v.detach().numpy())

			optimizer.zero_grad()
			loss_total.backward()
			optimizer.step()
			batch_idx += 1
		
		# print("Policy Loss", np.mean(losses_pi))
		# print("Value Loss", np.mean(losses_v))
		# print()
	return np.mean(losses_pi), np.mean(losses_v)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", action="store_true", default=False)
	parser.add_argument("--eval", action="store_true", default=False)
	parser.add_argument("--model", action="store", default="mlp", choices=["mlp", "resnet"])
	parser.add_argument("--n_iters", "-i", action="store", type=int, default=0)
	parser.add_argument("--n_episodes", "-s", action="store", type=int, default=0)
	parser.add_argument("--n_epochs", "-e", action="store", type=int, default=0)
	parser.add_argument("--n_searches", "-t", action="store", type=int, default=0)
	parser.add_argument("--n_matches", "-m", action="store", type=int, default=0)
	parser.add_argument("--n_processes", "-p", action="store", type=int, default=1)
	parser.add_argument("--batch", "-b", action="store", default=32, type=int)
	parser.add_argument("--render", "-r", action="store", choices=["tui", "human"], default=None)
	parser.add_argument("--device", "-d", action="store", default="cpu")
	parser.add_argument("--checkpoint", "-c", action="store", default="local/latest.pth")
	args = parser.parse_args()

	if (
		args.train and args.eval
	) or (
		not args.train and not args.eval
	):
		sys.exit("Train or eval?")
	if args.train and (
		not args.n_iters or
		not args.n_episodes or
		not args.n_epochs or
		not args.n_searches
	):
		sys.exit("Training requires --n_iters, --n_episodes, --n_epochs and --n_searches to be specified.")
	if args.eval and not args.n_matches:
		sys.exit("Evaluation requires --n_matches to be specified.")

	env = ultimatetictactoe.env(render_mode=args.render)
	if args.model == "mlp":
		model = MLP(torch.device(args.device))
	elif args.model == "resnet":
		model = ResNet(torch.device(args.device))
	else:
		sys.exit(f"Available models: mlp, resnet")
	if args.train:
		# _train(
		# 	env=env,
		# 	model=model,
		# 	n_iters=args.n_iters,
		# 	n_episodes=args.n_episodes,
		# 	n_epochs=args.n_epochs,
		# 	n_searches=args.n_searches,
		# 	batch_size=args.batch
		# )
		stats = _train_async(
			env_fn=lambda: ultimatetictactoe.env(render_mode=args.render),
			model=model,
			n_iters=args.n_iters,
			n_episodes=args.n_episodes,
			n_epochs=args.n_epochs,
			n_searches=args.n_searches,
			batch_size=args.batch,
			n_processes=args.n_processes,
			model_out=args.checkpoint
		)
		with open("local/training_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%s") + ".csv", "w") as f:
			f.write("loss_pi,loss_v\n")
			for line in stats:
				f.write(f"{line[0]},{line[1]}\n")
		# torch.save(model.state_dict(), args.checkpoint)
	elif args.eval:
		model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
		# wins, total = 0, 0
		# for _ in range(args.n_matches):
		# 	if _eval(env, model) > 0:
		# 		wins += 1
		# 	total += 1
		_eval(env, model, args.n_matches, args.n_searches)
		# print(f"Stats: model won {wins} out of {total} matches ({(wins / total) * 100}%)")