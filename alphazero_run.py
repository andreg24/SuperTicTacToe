import random
import torch

from ultimatetictactoe import ultimatetictactoe
from rl.alphazero.model import MLP
from rl.alphazero.mcts import MCTS
from rl.alphazero.utils import get_board_perspective


def episode(env: ultimatetictactoe.env, model):
	samples = []
	current_player = 1
	board = None
	mcts = MCTS(env)

	while True:
		state = get_board_perspective(env, current_player)
		root, action_probs = mcts.run(model, current_player, board)
		samples.append((state, current_player, action_probs))

		action, node, _ = root.select()
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

def train(env: ultimatetictactoe.env, model, n_iters, n_episodes):
	for i in range(1, n_iters + 1):
		print(f"Iteration {i}/{n_iters}")

		samples = []
		for j in range(1, n_episodes + 1):
			print(f"Episode {j}/{n_episodes}")
			samples.extend(episode(env, model))
		
		random.shuffle(samples)


if __name__ == "__main__":
	env = ultimatetictactoe.env(render_mode="human")
	model = MLP(torch.device("cpu"))
	train(env, model, 1, 1)

# if __name__ == "__main__":
# 	from rl.alphazero.mcts import MCTS
# 	from ultimatetictactoe import ultimatetictactoe as ttt

# 	env = ttt.env(render_mode=None)
# 	mcts = MCTS(env, n_searches=2048)
# 	mcts.run(model=None)