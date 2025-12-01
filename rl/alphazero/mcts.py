import numpy as np

from ultimatetictactoe import ultimatetictactoe
from .utils import _ucb, get_board_perspective

PERSPECTIVE_SELF = 1
PERSPECTIVE_OPPONENT = -1

def get_actions_value_prediction(env, model):
	# priors = np.ones(env.action_space(env.agent_selection).n)
	priors, value = model.predict(get_board_perspective(env, PERSPECTIVE_SELF))
	print(f"model says priors = {priors}, value = {value}")
	valid_moves = env.action_mask(env.agent_selection)
	priors *= valid_moves
	priors /= np.sum(priors)
	return {
		i: p for i, p in enumerate(priors)
	}, value


class Node:
	def __init__(self, prior, next_player, state=None, parent=None, C=1.414):
		self.prior = prior
		self.next_player = next_player
		self.state = state
		self.parent = parent
		self.children = {}
		self.count = 0
		self.value = 0

		self.C = C

	def has_children(self):
		return len(self.children) > 0

	def value(self):
		return self.value / self.count if self.count > 0 else 0

	def select(self):
		best = None, None, -np.inf
		for action, child in self.children.items():
			ucb = _ucb(self, child, self.C)
			if ucb > best[2]:
				best = action, child, ucb
		return best

	def expand(self, next_player, priors):
		for a, p in priors.items():
			if p > 0.0:
				self.children[a] = Node(prior=p, next_player=next_player, parent=self)

	def backpropagate(self, value, player):
		self.value += value if self.next_player == player else -value
		self.count += 1
		
		if self.parent != None:
			self.parent.backpropagate(value, player=self.next_player)


class MCTS:
	def __init__(self, env: ultimatetictactoe.raw_env, n_searches: int = 128):
		self.env = env
		self.env.reset()
		self.n_searches = n_searches

	def run(self, model, current_player=1, board=None):
		# Initialize the root node with a dummy prior and the current player.
		root = Node(
			prior=0.0,
			next_player=current_player,
			state=get_board_perspective(self.env, PERSPECTIVE_SELF)
		)
		priors, value = get_actions_value_prediction(self.env, model)
		root.expand(current_player * -1, priors)

		for _ in range(self.n_searches):
			self.env.reset(options=dict(board=board, next_player=current_player))

			# Start each search iteration from the root node and initialize
			# the traversal path with it.
			node = root

			# Traverse the tree until a leaf node is reached. For each non-leaf
			# node encountered, continue traversing the tree by selecting
			# one of its children according to their UCB score.
			while node.has_children():
				action, node, _ = node.select()
				# if not node.expanded():
				# 	print(f"playing action {action} as {self.env.agent_selection}")
				self.env.step(action)

			observation, reward, termination, truncation, info = self.env.last()

			value = -reward
			state_for_next = get_board_perspective(self.env, PERSPECTIVE_OPPONENT)
			if value == 0:
				priors, value = get_actions_value_prediction(self.env, model)
				node.state = state_for_next
				node.expand(node.next_player * -1, priors)

			node.backpropagate(value, node.next_player)

		action_probs = np.zeros(self.env.action_space(self.env.agent_selection).n)
		for a, child in root.children.items():
			action_probs[a] = child.count
		return root, action_probs / np.sum(action_probs)
