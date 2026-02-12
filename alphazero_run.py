import argparse
import random
import os
import sys
import numpy as np
import torch
import torch.optim as optim

from ultimatetictactoe import ultimatetictactoe
from rl.alphazero.model import MLP
from rl.alphazero.mcts import MCTS
from rl.alphazero.utils import get_board_perspective


def episode(env: ultimatetictactoe.env, model):
	samples = []
	current_player = 1
	board = None
	mcts = MCTS(env, n_searches=400)

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
				reward * (-1 ** (hist_player != current_player))
			) for hist_state, hist_player, hist_action_probs in samples]
		
		current_player *= -1
		board = env.board
		# print(f"No reward, doing another search starting from root -- {action} --> {node}")

def _train(env: ultimatetictactoe.env, model, n_iters, n_episodes, n_epochs, batch_size):
	for i in range(1, n_iters + 1):
		print(f"Iteration {i}/{n_iters}")

		samples = []
		for j in range(1, n_episodes + 1):
			print(f"Episode {j}/{n_episodes}")
			samples.extend(episode(env, model))
		
		random.shuffle(samples)
		print("All episodes executed. Training...")
		train_model(model, samples, n_epochs, batch_size)
		print()

def _eval(env: ultimatetictactoe.env, model):
	model.eval()

	mcts = MCTS(env, n_searches=400)
	current_player = 1
	board = None

	while True:
		if current_player == 1:
			action = env.action_space(env.agent_selection).sample(
				env.action_mask(env.agent_selection)
			)
			# print(f"Player {current_player} (random) playing action {action}")
		else:
			state = get_board_perspective(env, current_player)
			root, action_probs = mcts.run(model, current_player, board)
			action = np.argmax(action_probs)
			# print(f"Player {current_player} (model) playing action {action}")

		env.reset(options={
			"board": board,
			"next_player": current_player
		})
		env.step(action)
		observation, reward, termination, truncation, info = env.last()

		if termination:
			if reward > 0:
				print(f"Player {current_player} has won!")
			elif reward < 0:
				print(f"Player {current_player} has lost!")
			else:
				print("It's a tie!")
			return current_player * reward
		
		current_player *= -1
		board = env.board

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
			print(f"p_pi = {p_pi} | p_v = {p_v} | loss_pi = {loss_pi} | loss_v = {loss_v}")
			losses_pi.append(loss_pi.detach().numpy())
			losses_v.append(loss_v.detach().numpy())

			optimizer.zero_grad()
			loss_total.backward()
			optimizer.step()
			batch_idx += 1
		
		print("Policy Loss", np.mean(losses_pi))
		print("Value Loss", np.mean(losses_v))
		print()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", action="store_true", default=False)
	parser.add_argument("--eval", action="store_true", default=False)
	parser.add_argument("--model", action="store", default="mlp", choices=["mlp"])
	parser.add_argument("--n_iters", "-i", action="store", type=int, default=0)
	parser.add_argument("--n_episodes", "-s", action="store", type=int, default=0)
	parser.add_argument("--n_epochs", "-e", action="store", type=int, default=0)
	parser.add_argument("--n_matches", "-m", action="store", type=int, default=0)
	parser.add_argument("--batch", "-b", action="store", default=32, type=int)
	parser.add_argument("--render", "-r", action="store", choices=["tui", "human"], default=None)
	parser.add_argument("--device", "-d", action="store", default="cpu")
	parser.add_argument("--checkpoint", "-c", action="store", default="local/latest.pth")
	args = parser.parse_args()

	if args.train and args.eval:
		sys.exit("Train or eval?")
	if args.train and (
		not args.n_iters or
		not args.n_episodes or
		not args.n_epochs
	):
		sys.exit("Training requires --n_iters, --n_episodes and --n_epochs to be specified.")
	if args.eval and not args.n_matches:
		sys.exit("Evaluation requires --n_matches to be specified.")

	env = ultimatetictactoe.env(render_mode=args.render)
	model = MLP(torch.device(args.device))
	if args.train:
		_train(
			env=env,
			model=model,
			n_iters=args.n_iters,
			n_episodes=args.n_episodes,
			n_epochs=args.n_epochs,
			batch_size=args.batch
		)
		torch.save(model.state_dict(), args.checkpoint)
	elif args.eval:
		model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
		wins, total = 0, 0
		for _ in range(args.n_matches):
			if _eval(env, model) > 0:
				wins += 1
			total += 1
		print(f"Stats: model won {wins} out of {total} matches ({wins / total}%)")