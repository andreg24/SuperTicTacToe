"""
Watch a trained MinMaxQ agent play with pygame rendering.
"""

import argparse
import torch
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from SuperTicTacToe.ultimatetictactoe import ultimatetictactoe

from agent import MinMaxQAgent


class RandomAgent:
    name = "Random"
    device = torch.device('cpu')
    def pick_action(self, state, **kwargs):
        mask = state['action_mask']
        legal = np.where(mask)[0]
        action = np.random.choice(legal) if len(legal) > 0 else -1
        return {'action': action}


def play_visual_game(env, agent1, agent2, delay=0.5):
    env.reset()
    agents = {"player_1": agent1, "player_2": agent2}
    turn_count = 0
    print("\nGame started!")
    print(f"Player 1: {agent1.name}, Player 2: {agent2.name}")
    print("-" * 40)

    for agent_name in env.agent_iter():
        state, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if done:
            action = None
            print(f"\nGame Over! Turns: {turn_count}")
            print(f"Final rewards: {env.rewards}")
            if env.rewards["player_1"] > env.rewards["player_2"]:
                print(f"Winner: {agent1.name}")
            elif env.rewards["player_2"] > env.rewards["player_1"]:
                print(f"Winner: {agent2.name}")
            else:
                print("Draw!")
        else:
            agent = agents[agent_name]
            output = agent.pick_action(state)
            action = output['action']
            turn_count += 1
            print(f"Turn {turn_count}: {agent_name} plays {action}")
            time.sleep(delay)

        if isinstance(action, torch.Tensor):
            action = action.item()
        env.step(action)

    time.sleep(delay * 2)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = ultimatetictactoe.env(render_mode="human")

    # Load agent1
    if args.agent_path:
        agent1 = MinMaxQAgent("Trained", player_id=0, device=device, mode='eval')
        agent1.load(args.agent_path)
    else:
        print("No agent path provided, using random agent.")
        agent1 = RandomAgent()
        agent1.name = "Random1"

    # Create opponent
    if args.opponent == 'random':
        agent2 = RandomAgent()
        agent2.name = "Random"
    elif args.opponent == 'self':
        agent2 = agent1   # same agent
    else:
        agent2 = MinMaxQAgent("Untrained", player_id=1, device=device, mode='eval')

    play_visual_game(env, agent1, agent2, delay=args.delay)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str, help='Path to trained agent')
    parser.add_argument('--opponent', type=str, choices=['random', 'self', 'untrained'], default='random')
    parser.add_argument('--delay', type=float, default=0.5)
    args = parser.parse_args()
    main(args)