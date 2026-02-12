import math
import numpy as np

PERSPECTIVE_SELF = 1
PERSPECTIVE_OPPONENT = -1


def _ucb(parent, child, C):
	return (child.value / child.count) + C * math.sqrt(
		math.log(parent.count) / child.count
	) * child.prior if child.count > 0 else float("inf")

def get_board_perspective(env, perspective):
	current = env.agents.index(env.agent_selection) + 1
	opponent = 1 if current == 2 else 2
	board = np.array(env.board.cells)
	board = np.where(board == current, 1 if perspective == PERSPECTIVE_SELF else -1, board)
	board = np.where(board == opponent, -1 if perspective == PERSPECTIVE_SELF else 1, board)
	return board