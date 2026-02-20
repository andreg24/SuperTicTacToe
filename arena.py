import argparse

from rl.agent import AlphaZeroAgent, NeuralAgent
from rl.alphazero.model import MLP, MLP2, ResNet


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--agent1", "-a1", action="store", required=True)
	parser.add_argument("--agent2", "-a2", action="store", required=True)
	parser.add_argument("--checkpoint1", "-c1", action="store", required=True)
	parser.add_argument("--checkpoint2", "-c2", action="store", required=True)
	parser.add_argument("--options1", "-o1", action="store", required=False, default="")
	parser.add_argument("--options2", "-o2", action="store", required=False, default="")
	parser.add_argument("--n_matches", "-m", action="store", type=int, default=128)
	parser.add_argument("--device", "-d", action="store", default="cpu")
	args = parser.parse_args()

	agents = []
	all_agents = [{
		o[0]: o[1]
		for o in args.agent1.split(",")
	}, {
		o[0]: o[1]
		for o in args.agent2.split(",")
	}]
	all_options = [args.options1, args.options2]
	all_checkpoints = [args.checkpoint1, args.checkpoint2]
	for i, agent_type in enumerate(all_agents):
		if agent_type == "alphazero":
			options = all_agents[i]
			if options["model"] == "mlp1":
				model = MLP(torch.device(args.device))
			elif options["model"] == "mlp2":
				model = MLP2(torch.device(args.device))
			elif options["model"] == "resnet":
				model = ResNet(torch.device(args.device))
			model.load_state_dict(torch.load(all_checkpoints[i], weights_only=True))
			agent_fn = lambda env: AlphaZeroAgent(f"player_{i}", env, model, 2 * i - 1, n_searches=int(options["n_searches"]))
		elif agent_type == "ipg":
			options = all_agents[i]
			def load_agent(env):
				a = NeuralAgent(f"player_{i}")
				a.load(all_checkpoints[i])
			agent_fn = lambda env: NeuralAgent(f"player_{i}")