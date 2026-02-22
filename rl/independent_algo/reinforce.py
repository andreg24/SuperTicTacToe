# TODO

from typing import Optional
import os
import datetime
import tqdm
import random
import pickle
import numpy as np

import torch

from utils.board import TRANSFORMATIONS
from utils.board_utils import Rotation, Reflection
from .algo_utils import format_datetime
from rl.agent import RandomAgent, compute_games, Trajectory, NeuralAgent


def generate_schedule(n, px, pt, max_semi_turn=15):
    """Generates transformations schedule for n trajectories"""

    enable_transformation = np.random.binomial(1, px, (n,))
    use_transformation = np.random.randint(0, 5, (n,)) * enable_transformation
    transformation_turns = (
        np.random.binomial(max_semi_turn, pt, (n,)) * enable_transformation * 2
    )
    return use_transformation, transformation_turns


# manipolate trajectory
def reinforce_update(agent, trajectory, gamma=0.99):
    rewards = trajectory["rewards"]
    log_probs = trajectory["log_probs"]
    log_probs_tensor = torch.cat(log_probs)
    G = torch.zeros(1)
    n = len(trajectory["observations"])
    returns = []
    for i in range(n - 1, -1, -1):
        # G = rewards[i] + G * gamma ** (i - n + 1)
        G = rewards[i] + G * gamma
        returns.insert(0, G.item())
    returns = torch.tensor(returns, dtype=torch.float32, device=log_probs_tensor.device)

    # loss = -(G * log_probs_tensor).sum()
    loss = -(returns * log_probs_tensor).sum()
    agent.update(loss)
    return loss.item()


def reinforce(
    env,
    agent_1,
    agent_2,
    num_episodes,
    gamma=0.99,
    update1=True,
    update2=True,
    enable_swap=False,
    enable_transform=False,
    px=0.3,
    pt=0.5,
    max_semi_turn=15,
    checkpoint_rate: Optional[int] = None,
    experiment_name: Optional[str] = None,
    validation_rate: Optional[int] = None,
    weights_path: Optional[str] = "",
    weights_name: Optional[str] = "",
    device: torch.device = torch.device("cpu")
) -> None:
    if (num_episodes-1)%100 != 0:
        num_episodes = num_episodes//100 * 100 + 101
    
    agent_1_losses = []
    agent_2_losses = []

    starting_epoch_name = 0
    if weights_path != "":
        starting_epoch_name = int(weights_name.split("_")[1].split(".")[0])
        agent_1.policy_net.load_state_dict(torch.load(weights_path + "/agent_1/" + weights_name))
        agent_2.policy_net.load_state_dict(torch.load(weights_path + "/agent_2/" + weights_name))
    
    if checkpoint_rate is not None:
        # save parameters
        param_values = locals().copy()
        # remove objects you don't want to serialize/store directly
        param_values.pop("agent_1", None)
        param_values.pop("agent_2", None)
        param_values.pop("env", None)
        param_values.pop("agent_1_losses", None)
        param_values.pop("agent_2_losses", None)
        param_values.pop("experiment_name", None)

        datetime_string = format_datetime(str(datetime.datetime.now()))
        if experiment_name is None:
            experiment_name = datetime_string
        # check folder existence
        checkpoint_folder = f"./rl/independent_algo/logs/checkpoints/{experiment_name}/{datetime_string}"
        if update1:
            os.makedirs(checkpoint_folder + "/agent_1", exist_ok=True)
        if update2:
            os.makedirs(checkpoint_folder + "/agent_2", exist_ok=True)


        # add selected agent hyperparameters
        for prefix, agent in [("agent_1", agent_1), ("agent_2", agent_2)]:
            param_values.update({
                f"{prefix}_epsilon": agent.epsilon,
                f"{prefix}_learning_power": agent.learning_power,
                f"{prefix}_learning_const": agent.learning_const,
                f"{prefix}_exploration_power": agent.exploration_power,
                f"{prefix}_exploration_const": agent.exploration_const,
            })
        param_values.update({
            "agent_1": str(agent_1.policy_net),
            "agent_2": str(agent_2.policy_net)
        })
        
        with open(checkpoint_folder+'/params.pkl', 'wb') as file:
            pickle.dump(param_values, file)


    
    if validation_rate is not None:
        res_12 = []
        res_1r = []
        res_2r = []
        ar = RandomAgent("boh")
    else:
        res_12 = None
        res_1r = None
        res_2r = None


    TR = Trajectory(env, agent_1, agent_2, True, enable_tansform=enable_transform)
    if enable_transform:
        schedule = generate_schedule(num_episodes, px, pt, max_semi_turn)

    for ep in tqdm.trange(num_episodes):
        if update1:
            agent_1.train()
        if update2:
            agent_2.train()

        if isinstance(agent_2, NeuralAgent):
            agent_1.policy_net.disable_epsilon = False
        if isinstance(agent_2, NeuralAgent):
            agent_2.policy_net.disable_epsilon = False


        # swap players at each epoch
        if enable_swap and ep > 0:
            TR.swap_players()

        # perform transformation
        if enable_transform and schedule[1][ep] != 0:
            TR.compute(
                schedule[1][ep], TRANSFORMATIONS[schedule[0][ep]], schedule[1][ep]
            )
        else:
            TR.compute()

        trajectory = TR.trajectory

        if not enable_swap or ep % 2 == 0:
            if update1:
                loss = reinforce_update(agent_1, trajectory["player_1"], gamma)
                agent_1_losses.append(loss)
            if update2:
                loss = reinforce_update(agent_2, trajectory["player_2"], gamma)
                agent_2_losses.append(loss)
        else:
            if update1:
                loss = reinforce_update(agent_1, trajectory["player_2"], gamma)
                agent_1_losses.append(loss)
            if update2:
                loss = reinforce_update(agent_2, trajectory["player_1"], gamma)
                agent_2_losses.append(loss)
        
        if checkpoint_rate is not None:
            if ep != 0 and ep%checkpoint_rate == 0 or ep==num_episodes-1:
                # save weights
                if update1:
                    torch.save(agent_1.policy_net.state_dict(), checkpoint_folder+f"/agent_1/model_{ep+starting_epoch_name}.pt")
                if update2:
                    torch.save(agent_2.policy_net.state_dict(), checkpoint_folder+f"/agent_2/model_{ep+starting_epoch_name}.pt")
        
        if validation_rate is not None:
            if ep != 0 and ep%validation_rate == 0 or ep == num_episodes-1:
                res_12.append(compute_games(env, agent1=agent_1, agent2=agent_2, n=200, verbose=False))
                res_1r.append(compute_games(env, agent1=agent_1, agent2=ar, n=200, verbose=False))
                res_2r.append(compute_games(env, agent1=agent_2, agent2=ar, n=200, verbose=False))
            
            res = (agent_1_losses, agent_2_losses, res_12, res_1r, res_2r)
            with open(checkpoint_folder+'/res.pkl', 'wb') as file:
                pickle.dump(res, file)

    return agent_1_losses, agent_2_losses, res_12, res_1r, res_2r
