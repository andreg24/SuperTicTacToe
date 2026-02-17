"""
Watch a trained MinMaxQ agent play with pygame rendering.
"""

import argparse
import torch
import time
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ultimatetictactoe import ultimatetictactoe
from agent import MinMaxQAgent


class RandomAgent:
    """Simple random agent for testing."""
    name = "Random"
    device = torch.device('cpu')
    
    def pick_action(self, state, **kwargs):
        mask = state['action_mask']
        legal = np.where(mask)[0]
        action = np.random.choice(legal) if len(legal) > 0 else -1
        return {'action': action}


def play_visual_game(env, agent1, agent2, delay=0.5):
    """Play one game with visual rendering."""
    env.reset()
    agents = {"player_1": agent1, "player_2": agent2}
    turn_count = 0
    
    print("\n" + "="*60)
    print("GAME STARTED!")
    print("="*60)
    print(f"Player 1 (X): {agent1.name}")
    print(f"Player 2 (O): {agent2.name}")
    print("-" * 60)
    
    for agent_name in env.agent_iter():
        state, reward, termination, truncation, info = env.last()
        done = termination or truncation
        
        if done:
            action = None
            print("\n" + "="*60)
            print("GAME OVER!")
            print("="*60)
            print(f"Total turns: {turn_count}")
            print(f"Final rewards: {env.rewards}")
            
            if env.rewards["player_1"] > env.rewards["player_2"]:
                print(f"🏆 Winner: {agent1.name} (Player 1)")
            elif env.rewards["player_2"] > env.rewards["player_1"]:
                print(f"🏆 Winner: {agent2.name} (Player 2)")
            else:
                print("🤝 Draw!")
            print("="*60)
        else:
            agent = agents[agent_name]
            output = agent.pick_action(state)
            action = output['action']
            turn_count += 1
            
            player_symbol = "X" if agent_name == "player_1" else "O"
            print(f"Turn {turn_count:2d}: {agent_name:8s} ({player_symbol}) plays position {action:2d}")
            time.sleep(delay)
        
        if isinstance(action, torch.Tensor):
            action = action.item()
        
        env.step(action)
    
    time.sleep(delay * 2)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = ultimatetictactoe.env(render_mode="human")
    
    # Load agent1
    if args.agent_path:
        print(f"Loading trained agent from: {args.agent_path}")
        agent1 = MinMaxQAgent("Trained Agent", player_id=0, device=device, mode='eval')
        agent1.load(args.agent_path)
    else:
        print("No agent path provided, using random agent.")
        agent1 = RandomAgent()
        agent1.name = "Random Agent 1"
    
    # Create opponent
    if args.opponent == 'random':
        agent2 = RandomAgent()
        agent2.name = "Random Agent"
    elif args.opponent == 'self':
        agent2 = agent1   # same agent plays against itself
        agent2.name = agent1.name + " (copy)"
    else:
        agent2 = MinMaxQAgent("Untrained Agent", player_id=1, device=device, mode='eval')
    
    # Play game
    play_visual_game(env, agent1, agent2, delay=args.delay)
    
    # Cleanup
    env.close()
    print("\nDemo finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo MinMaxQ agent with visual rendering")
    parser.add_argument('--agent_path', type=str, help='Path to trained agent weights')
    parser.add_argument('--opponent', type=str, choices=['random', 'self', 'untrained'], 
                       default='random', help='Type of opponent')
    parser.add_argument('--delay', type=float, default=0.5, 
                       help='Delay between moves in seconds')
    
    args = parser.parse_args()
    main(args)