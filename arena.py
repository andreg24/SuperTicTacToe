import argparse
import json
import torch
import sys
import datetime
import numpy as np
from copy import copy

from rl.agent import AlphaZeroAgent, NeuralAgent, RandomAgent, async_compute_games
from rl.minmaxq.agent import MinMaxQAgent
from rl.alphazero.model import MLP, MLP2, ResNet
from ultimatetictactoe import ultimatetictactoe


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--agent1", "-a1", action="store", required=True)
	parser.add_argument("--agent2", "-a2", action="store", required=True)
	parser.add_argument("--checkpoint1", "-c1", action="store", required=False)
	parser.add_argument("--checkpoint2", "-c2", action="store", required=False)
	parser.add_argument("--options1", "-o1", action="store", required=False, default="")
	parser.add_argument("--options2", "-o2", action="store", required=False, default="")
	parser.add_argument("--n_matches", "-m", action="store", type=int, default=128)
	parser.add_argument("--n_processes", "-p", action="store", type=int, default=1)
	parser.add_argument("--temperature", "-t", action="store", type=float, default=1.0)
	parser.add_argument("--device", "-d", action="store", default="cpu")
	args = parser.parse_args()

	agents = []
	all_options = [{
		o.split("=")[0]: o.split("=")[1]
		for o in args.options1.split(",")
	} if args.options1 else {}, {
		o.split("=")[0]: o.split("=")[1]
		for o in args.options2.split(",")
	} if args.options2 else {}]
	all_agents = [args.agent1, args.agent2]
	all_checkpoints = [args.checkpoint1, args.checkpoint2]
	for i, agent_type in enumerate(all_agents):
		if agent_type == "a0":
			options = all_options[i]
			if options["model"] == "mlp1":
				model = MLP(torch.device(args.device))
			elif options["model"] == "mlp2":
				model = MLP2(torch.device(args.device))
			elif options["model"] == "resnet":
				model = ResNet(torch.device(args.device))
			model.load_state_dict(torch.load(all_checkpoints[i], weights_only=True))
			def f(env, i=i, model=model, options=options):
				n_searches = int(options["n_searches"])
				return AlphaZeroAgent(f"player_{i}", env, model, -1 if i == 1 else 1, n_searches=n_searches)
			agent_fn = f
		elif agent_type == "ipg":
			def load_agent(env, i=i, checkpoint=all_checkpoints[i]):
				a = NeuralAgent(f"player_{i}")
				a.load(checkpoint)
				a.disable_epsilon(True)
				return a
			agent_fn = load_agent
		elif agent_type == "mmq":
			def load_agent(env, i=i, checkpoint=all_checkpoints[i]):
				a = MinMaxQAgent(f"player_{i}", i + 1)
				a.load(checkpoint)
				return a
			agent_fn = load_agent
		elif agent_type == "rnd":
			agent_fn = lambda env: RandomAgent(f"player_{i}", action_mask_enabled=True)
		else:
			sys.exit(f"Invalid agent type {agent_type}")
		agents.append(agent_fn)
	
	env_fn = lambda: ultimatetictactoe.env(render_mode=None)
	stats = async_compute_games(
		env_fn=env_fn,
		agent1_fn=agents[0],
		agent2_fn=agents[1],
		n_games=args.n_matches,
		n_processes=args.n_processes,
		temperature=args.temperature,
		enable_swap=False,
		verbose=False
	)

	stats["agent1"] = all_agents[0]
	stats["agent2"] = all_agents[1]
	stats["checkpoint1"] = all_checkpoints[0]
	stats["checkpoint2"] = all_checkpoints[1]
	stats["options1"] = all_options[0]
	stats["options2"] = all_options[1]

	class NpEncoder(json.JSONEncoder):
		def default(self, obj):
			if isinstance(obj, np.integer):
				return int(obj)
			if isinstance(obj, np.floating):
				return float(obj)
			if isinstance(obj, np.ndarray):
				return obj.tolist()
			return super(NpEncoder, self).default(obj)

	with open(f"local/arena_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
		json.dump(stats, f, cls=NpEncoder)