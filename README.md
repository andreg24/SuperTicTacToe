# UltimateTicTacToe

Three RL approaches for playing UltimateTicTacToe:
- Independent symmetric DQN
- Independent asymmetric REINFORCE [Independent Policy Gradient Methods for Competitive Reinforcement Learning](https://arxiv.org/abs/2101.04233)
- AlphaZero

# AlphaZero
An AlphaZero implementation is provided in the `rl/alphazero` folder. The MCTS algorithm is implemented in `mcts.py`, while NN models are defined as `torch.nn.Module` classes in `models.py`. The `ResNet` model has been adapted from the original paper, while `MLP` and `MLP2` are simpler architectures, the latter of which provided satisfactory results. It is supposed to embed te state of the smaller sub-boards first, then aggregating the embeddings together, separating a policy head (outputting a value for each possible action) and a value head (generating a single value in [-1, 1] as the game outcome of the given state).

Functions for performing (a)sync training and evaluation of the algorithm are provided in `alphazero_run.py`, which can be invoked as a script, with self-explanatory arguments. It can be tested using

```bash
$ python alphazero_run.py [--train | --eval] -i <number of overall iterations> -e <number of training epochs for each iteration> -t <number of MCTS searches per episode> -s <number of episodes per iteration> -p <number of parallel processes> [-c <checkpoint file>] [--save] [--save_intermediate]
```