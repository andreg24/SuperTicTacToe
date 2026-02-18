#!/usr/bin/env python3
"""
Unified Evaluation Script for Ultimate Tic-Tac-Toe RL Algorithms
Uses compute_games() from rl/independent_algo/reinforce.py

Evaluates:
- REINFORCE agents
- MinMaxQ agents  
- AlphaZero agents

All against RandomAgent baseline
"""

import sys
import os
import torch
import argparse
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ultimatetictactoe.ultimatetictactoe import raw_env
from pettingzoo.utils import wrappers
from rl.agent import RandomAgent, NeuralAgent
from rl.minmaxq import MinMaxQAgent
from rl.alphazero.model import MLP, ResNet
from rl.independent_algo.reinforce import compute_games


def create_env():
    """Create environment with proper wrappers (matches original env() function)"""
    env = raw_env(render_mode=None)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


# ============================================================
# WRAPPER PER MINMAXQ (BaseAgent compatible)
# ============================================================
class MinMaxQWrapper:
    """Wrapper to make MinMaxQ compatible with compute_games"""
    
    def __init__(self, minmaxq_agent):
        self.agent = minmaxq_agent
        self.name = minmaxq_agent.name
        
    def pick_action(self, state):
        """Wrap MinMaxQ's pick_action to match BaseAgent interface"""
        output = self.agent.pick_action(state)
        return {'action': output['action']}
    
    def eval(self):
        """Set to eval mode"""
        self.agent.mode = 'eval'


# ============================================================
# WRAPPER PER ALPHAZERO (BaseAgent compatible)
# ============================================================
class AlphaZeroWrapper:
    """Wrapper to make AlphaZero compatible with compute_games"""
    
    def __init__(self, alphazero_agent):
        self.agent = alphazero_agent
        self.name = alphazero_agent.name
        
    def pick_action(self, state):
        """Wrap AlphaZero's pick_action to match BaseAgent interface"""
        output = self.agent.pick_action(state)
        return {'action': output['action']}
    
    def eval(self):
        """Set to eval mode"""
        self.agent.model.eval()


# ============================================================
# LOAD AGENTS
# ============================================================
def load_reinforce_agent(path: str, device: torch.device):
    """Load REINFORCE agent"""
    print(f"Loading REINFORCE agent from {path}...")
    agent = NeuralAgent(
        "REINFORCE",
        epsilon=0.0,  # No exploration during eval
        device=device,
        mode='sample'  # or 'argmax'
    )
    agent.policy_net.load_state_dict(torch.load(path, map_location=device))
    agent.eval()
    return agent


def load_minmaxq_agent(path: str, device: torch.device):
    """Load MinMaxQ agent"""
    print(f"Loading MinMaxQ agent from {path}...")
    agent = MinMaxQAgent(
        "MinMaxQ",
        player_id=0,
        device=device,
        mode='eval'
    )
    agent.load(path)
    return MinMaxQWrapper(agent)


def load_alphazero_agent(path: str, model_type: str, device: torch.device, n_searches: int = 128):
    """Load AlphaZero agent"""
    print(f"Loading AlphaZero ({model_type}) agent from {path}...")
    
    # Create appropriate model
    if model_type == 'mlp':
        model = MLP(device)
    elif model_type == 'resnet':
        model = ResNet(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    
    # Create AlphaZero agent (this needs the env)
    # We'll create it lazily in evaluate_agent
    return {'model': model, 'n_searches': n_searches}


# ============================================================
# EVALUATION
# ============================================================
def evaluate_agent(env, agent, agent_type: str, n_games: int = 1024):
    """
    Evaluate agent vs RandomAgent using compute_games
    
    Args:
        env: Environment
        agent: Agent to evaluate (BaseAgent compatible)
        agent_type: Type of agent ('reinforce', 'minmaxq', 'alphazero')
        n_games: Number of games to play
    
    Returns:
        dict: Results from compute_games
    """
    random_agent = RandomAgent("Random", action_mask_enabled=True)
    
    # For AlphaZero, need to instantiate the agent with env
    if agent_type == 'alphazero':
        from rl.agent import AlphaZeroAgent
        alphazero_agent = AlphaZeroAgent(
            "AlphaZero",
            env,
            agent['model'],
            player=-1,  # Player 2 perspective
            n_searches=agent['n_searches']
        )
        agent = AlphaZeroWrapper(alphazero_agent)
    
    print(f"\n{'='*60}")
    print(f"Evaluating {agent.name} vs Random ({n_games} games)...")
    print(f"{'='*60}\n")
    
    results = compute_games(
        env,
        agent,
        random_agent,
        n=n_games,
        enable_swap=True  # Alternate who plays first
    )
    
    return results


def print_results(agent_name: str, results: dict):
    """Pretty print evaluation results"""
    print(f"\n{'='*70}")
    print(f"RESULTS: {agent_name} vs Random")
    print(f"{'='*70}")
    print(f"  Win Rate:   {results['results'][0]:.2f}%")
    print(f"  Loss Rate:  {results['results'][1]:.2f}%")
    print(f"  Draw Rate:  {results['results'][2]:.2f}%")
    print(f"  Avg Reward: {results['rewards'].mean():.4f}")
    print(f"  Avg Turns:  {results['game_turns'].mean():.1f}")
    print(f"{'='*70}")
    
    # Detailed reward breakdown
    print("\nReward Distribution:")
    for reward, count in sorted(results['rewards_count'].items()):
        print(f"  {reward:+.1f}: {count:.2f}%")


# ============================================================
# MAIN
# ============================================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create environment
    env = create_env()
    
    all_results = {}
    
    # ============================================================
    # Evaluate REINFORCE
    # ============================================================
    if args.reinforce_path:
        agent = load_reinforce_agent(args.reinforce_path, device)
        results = evaluate_agent(env, agent, 'reinforce', args.n_games)
        print_results("REINFORCE", results)
        all_results['REINFORCE'] = results
    
    # ============================================================
    # Evaluate MinMaxQ
    # ============================================================
    if args.minmaxq_path:
        agent = load_minmaxq_agent(args.minmaxq_path, device)
        results = evaluate_agent(env, agent, 'minmaxq', args.n_games)
        print_results("MinMaxQ", results)
        all_results['MinMaxQ'] = results
    
    # ============================================================
    # Evaluate AlphaZero
    # ============================================================
    if args.alphazero_path:
        agent_dict = load_alphazero_agent(
            args.alphazero_path,
            args.alphazero_model,
            device,
            args.alphazero_searches
        )
        results = evaluate_agent(env, agent_dict, 'alphazero', args.n_games)
        print_results(f"AlphaZero ({args.alphazero_model})", results)
        all_results['AlphaZero'] = results
    
    # ============================================================
    # Summary Comparison
    # ============================================================
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY COMPARISON")
        print(f"{'='*70}")
        print(f"{'Algorithm':<20} {'Win %':<12} {'Loss %':<12} {'Draw %':<12} {'Avg Turns'}")
        print("-" * 70)
        
        for name, res in all_results.items():
            print(f"{name:<20} {res['results'][0]:>10.2f}% {res['results'][1]:>10.2f}% "
                  f"{res['results'][2]:>10.2f}% {res['game_turns'].mean():>12.1f}")
        print(f"{'='*70}\n")
    
    # ============================================================
    # Save results (optional)
    # ============================================================
    if args.save_results:
        import json
        output_file = Path(args.save_results)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for name, res in all_results.items():
            serializable_results[name] = {
                'win_rate': float(res['results'][0]),
                'loss_rate': float(res['results'][1]),
                'draw_rate': float(res['results'][2]),
                'avg_reward': float(res['rewards'].mean()),
                'avg_turns': float(res['game_turns'].mean()),
                'reward_distribution': {float(k): float(v) for k, v in res['rewards_count'].items()}
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for all RL algorithms"
    )
    
    # Agent paths
    parser.add_argument('--reinforce_path', type=str, help='Path to REINFORCE weights (.pt file)')
    parser.add_argument('--minmaxq_path', type=str, help='Path to MinMaxQ weights (.pt file)')
    parser.add_argument('--alphazero_path', type=str, help='Path to AlphaZero weights (.pt file)')
    parser.add_argument('--alphazero_model', type=str, default='mlp', choices=['mlp', 'resnet'], help='AlphaZero model type')
    parser.add_argument('--alphazero_searches', type=int, default=128,
                       help='Number of MCTS searches for AlphaZero')
    
    # Evaluation settings
    parser.add_argument('--n_games', type=int, default=1024, help='Number of games to play per agent')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    # Output
    parser.add_argument('--save_results', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Validate that at least one agent is specified
    if not any([args.reinforce_path, args.minmaxq_path, args.alphazero_path]):
        parser.error("At least one agent path must be specified")
    
    main(args)