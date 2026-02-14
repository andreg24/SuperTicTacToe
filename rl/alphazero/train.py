import random
import torch
from multiprocessing import Pool

from ..ultimatetictactoe import ultimatetictactoe
from .model import MLP
from .mcts import MCTS
from .utils import get_board_perspective


def episode(env: ultimatetictactoe.env, model):
	samples = []
	current_player = 1
	board = None
	mcts = MCTS(env)

	while True:
		state = get_board_perspective(env, current_player)
		root, action_probs = mcts.run(model, current_player, board)
		samples.append(
			(state, current_player, action_probs)
		)

		action, node = root.select()
		env.step(action)
		observation, reward, termination, truncation, info = env.last()
		reward = -reward

		if reward != 0:
			return [(
				hist_state,
				hist_action_probs,
				reward * (-1 ** (hist_player != current_player))
			) for hist_state, hist_player, hist_action_probs in samples]
		
		current_player *= -1
		board = env.board.copy()

def episode_async(env_fn: callable, model):
	env = env_fn()
	return episode(env, model)

def train(env: ultimatetictactoe.env, model, n_iters, n_episodes):
	for i in range(1, n_iters + 1):
		print(f"{i}/{n_iters}")

		samples = []
		for _ in range(n_episodes):
			samples.extend(episode(env, model))
		
		random.shuffle(samples)

def train_async(env_fn: callable, model, n_iters, n_episodes):
	for i in range(1, n_iters + 1):
		print(f"{i}/{n_iters}")

		with Pool(processes=8) as pool:
			results = pool.starmap(episode_async, [(env_fn, model) for _ in range(n_episodes)])
		
		samples = []
		for ep_samples in results:
			samples.extend(ep_samples)

		random.shuffle(samples)

if __name__ == "__main__":
	env = ultimatetictactoe.env()
	model = MLP(torch.device("cpu"))
	train(env, model, 128, 128)