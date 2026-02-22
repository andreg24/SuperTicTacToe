"""
Comprehensive evaluation script for trained MinMaxQ agent.

Usage:
    python evaluate_trained_agent.py --model_path agent1.pt --num_games 100
    
Evaluates agent against random opponent and provides:
- Win/Loss/Draw rates
- Average reward
- Average number of turns
- Turn statistics by outcome
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from SuperTicTacToe.ultimatetictactoe import ultimatetictactoe
from agent import MinMaxQAgent


class RandomAgent:
    """Simple random agent for baseline comparison."""
    def __init__(self):
        self.name = "Random"
        self.device = torch.device('cpu')
    
    def pick_action(self, state, **kwargs):
        mask = state['action_mask']
        legal = np.where(mask)[0]
        if len(legal) > 0:
            action = np.random.choice(legal)
        else:
            action = -1
        return {'action': action, 'epsilon': 0.0}


def evaluate_agent_detailed(env, agent, num_games=100, verbose=True):
    """
    Comprehensive evaluation of agent vs random opponent.
    
    Returns:
        dict with detailed statistics:
        - win_rate, loss_rate, draw_rate (%)
        - avg_reward_agent, avg_reward_random
        - avg_turns_all, avg_turns_win, avg_turns_loss, avg_turns_draw
        - all_rewards, all_turns (raw data)
    """
    random_agent = RandomAgent()
    
    # Set agent to evaluation mode
    original_mode = agent.mode
    agent.mode = 'eval'
    
    # Statistics collectors
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'rewards_agent': [],
        'rewards_random': [],
        'turns_all': [],
        'turns_win': [],
        'turns_loss': [],
        'turns_draw': [],
        'first_moves': [],
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating {agent.name} vs Random Agent")
        print(f"Number of games: {num_games}")
        print(f"{'='*60}\n")
    
    for game_idx in range(num_games):
        env.reset()
        action_count = 0  # Count individual actions
        turn_count = 0    # Count complete turns (both players move)
        first_move = None
        
        agents_dict = {"player_1": agent, "player_2": random_agent}
        
        for agent_name in env.agent_iter():
            state, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            if done:
                # Record final rewards
                agent_reward = env.rewards.get('player_1', 0)
                random_reward = env.rewards.get('player_2', 0)
                
                results['rewards_agent'].append(agent_reward)
                results['rewards_random'].append(random_reward)
                results['turns_all'].append(turn_count)
                
                # Categorize outcome
                if agent_reward > random_reward:
                    results['wins'] += 1
                    results['turns_win'].append(turn_count)
                elif random_reward > agent_reward:
                    results['losses'] += 1
                    results['turns_loss'].append(turn_count)
                else:
                    results['draws'] += 1
                    results['turns_draw'].append(turn_count)
                
                break
            
            current_agent = agents_dict[agent_name]
            output = current_agent.pick_action(state)
            action = output['action']
            
            # Record first move
            if action_count == 0 and agent_name == 'player_1':
                first_move = action
                results['first_moves'].append(action)
            
            if isinstance(action, torch.Tensor):
                action = action.item()
            
            env.step(action)
            action_count += 1
            
            # Increment turn count after player_2's move (complete turn)
            if agent_name == 'player_2':
                turn_count += 1
    
    # Restore agent mode
    agent.mode = original_mode
    
    # Compute statistics
    total_games = results['wins'] + results['losses'] + results['draws']
    
    stats = {
        # Win/Loss/Draw rates
        'win_rate': results['wins'] / total_games * 100,
        'loss_rate': results['losses'] / total_games * 100,
        'draw_rate': results['draws'] / total_games * 100,
        
        # Average rewards
        'avg_reward_agent': np.mean(results['rewards_agent']),
        'avg_reward_random': np.mean(results['rewards_random']),
        'std_reward_agent': np.std(results['rewards_agent']),
        'std_reward_random': np.std(results['rewards_random']),
        
        # Average turns (all games)
        'avg_turns_all': np.mean(results['turns_all']),
        'std_turns_all': np.std(results['turns_all']),
        'min_turns': np.min(results['turns_all']),
        'max_turns': np.max(results['turns_all']),
        
        # Average turns by outcome
        'avg_turns_win': np.mean(results['turns_win']) if results['turns_win'] else None,
        'avg_turns_loss': np.mean(results['turns_loss']) if results['turns_loss'] else None,
        'avg_turns_draw': np.mean(results['turns_draw']) if results['turns_draw'] else None,
        
        # Strategic metrics
        'first_move_center_rate': (np.array(results['first_moves']) == 40).mean() * 100 if results['first_moves'] else 0,
        
        # Raw data
        'all_rewards_agent': results['rewards_agent'],
        'all_rewards_random': results['rewards_random'],
        'all_turns': results['turns_all'],
        
        # Counts
        'total_games': total_games,
        'wins': results['wins'],
        'losses': results['losses'],
        'draws': results['draws'],
    }
    
    return stats


def print_evaluation_summary(stats, verbose=True):
    """Print formatted evaluation summary."""
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    print(f"📊 GAME OUTCOMES (n={stats['total_games']} games)")
    print(f"  Wins:   {stats['wins']:4d} ({stats['win_rate']:5.1f}%)")
    print(f"  Losses: {stats['losses']:4d} ({stats['loss_rate']:5.1f}%)")
    print(f"  Draws:  {stats['draws']:4d} ({stats['draw_rate']:5.1f}%)")
    
    print(f"\n💰 AVERAGE REWARDS")
    print(f"  Agent:  {stats['avg_reward_agent']:+.3f} ± {stats['std_reward_agent']:.3f}")
    print(f"  Random: {stats['avg_reward_random']:+.3f} ± {stats['std_reward_random']:.3f}")
    
    print(f"\n⏱️  AVERAGE NUMBER OF TURNS")
    print(f"  All games:  {stats['avg_turns_all']:.1f} ± {stats['std_turns_all']:.1f} turns")
    print(f"  Range:      {stats['min_turns']:.0f} - {stats['max_turns']:.0f} turns")
    
    print(f"\n📈 TURNS BY OUTCOME")
    if stats['avg_turns_win'] is not None:
        print(f"  When winning:  {stats['avg_turns_win']:.1f} turns (avg)")
    if stats['avg_turns_loss'] is not None:
        print(f"  When losing:   {stats['avg_turns_loss']:.1f} turns (avg)")
    if stats['avg_turns_draw'] is not None:
        print(f"  When drawing:  {stats['avg_turns_draw']:.1f} turns (avg)")
    
    print(f"\n🎯 STRATEGIC BEHAVIOR")
    print(f"  First move center: {stats['first_move_center_rate']:.1f}%")
    print(f"  (Random baseline: 1.23%)")
    
    print(f"\n{'='*60}\n")
    
    # Interpretation
    if stats['win_rate'] > 70:
        print("✅ EXCELLENT: Agent performs significantly above random baseline!")
    elif stats['win_rate'] > 50:
        print("✓ GOOD: Agent outperforms random opponent.")
    elif stats['win_rate'] > 40:
        print("⚠️  FAIR: Agent slightly better than random.")
    else:
        print("❌ POOR: Agent needs more training.")
    
    if stats['avg_turns_win'] and stats['avg_turns_loss']:
        if stats['avg_turns_win'] < stats['avg_turns_loss']:
            print("✅ Agent plays efficiently: wins quickly, loses slowly (defensive)")
        else:
            print("⚠️  Agent wins take longer than losses")
    
    print(f"\n{'='*60}\n")


def save_results_to_file(stats, save_path):
    """Save detailed results to text file."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MinMaxQ Agent Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total games: {stats['total_games']}\n\n")
        
        f.write("OUTCOMES:\n")
        f.write(f"  Wins:   {stats['wins']} ({stats['win_rate']:.2f}%)\n")
        f.write(f"  Losses: {stats['losses']} ({stats['loss_rate']:.2f}%)\n")
        f.write(f"  Draws:  {stats['draws']} ({stats['draw_rate']:.2f}%)\n\n")
        
        f.write("REWARDS:\n")
        f.write(f"  Agent avg:  {stats['avg_reward_agent']:+.4f} ± {stats['std_reward_agent']:.4f}\n")
        f.write(f"  Random avg: {stats['avg_reward_random']:+.4f} ± {stats['std_reward_random']:.4f}\n\n")
        
        f.write("TURNS:\n")
        f.write(f"  Overall avg: {stats['avg_turns_all']:.2f} ± {stats['std_turns_all']:.2f}\n")
        f.write(f"  Range: {stats['min_turns']:.0f} - {stats['max_turns']:.0f}\n")
        if stats['avg_turns_win']:
            f.write(f"  When winning: {stats['avg_turns_win']:.2f}\n")
        if stats['avg_turns_loss']:
            f.write(f"  When losing: {stats['avg_turns_loss']:.2f}\n")
        if stats['avg_turns_draw']:
            f.write(f"  When drawing: {stats['avg_turns_draw']:.2f}\n")
        
        f.write(f"\nSTRATEGIC:\n")
        f.write(f"  First move center: {stats['first_move_center_rate']:.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Results saved to: {save_path}")


def main(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load environment
    env = ultimatetictactoe.env()
    print("Environment created.")
    
    # Load agent
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    agent = MinMaxQAgent(
        name="MinMaxQ_Agent",
        player_id=0,
        device=device,
        mode='eval'
    )
    agent.load(str(model_path))
    print("✓ Model loaded successfully")
    
    # Run evaluation
    stats = evaluate_agent_detailed(
        env=env,
        agent=agent,
        num_games=args.num_games,
        verbose=args.verbose
    )
    
    # Print results
    print_evaluation_summary(stats, verbose=args.verbose)
    
    # Save results
    if args.save_results:
        save_path = model_path.parent / f"evaluation_results_{args.num_games}games.txt"
        save_results_to_file(stats, save_path)
    
    # Return stats for programmatic use
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained MinMaxQ agent")
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model (e.g., agent1.pt)'
    )
    
    parser.add_argument(
        '--num_games',
        type=int,
        default=100,
        help='Number of evaluation games (default: 100)'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage'
    )
    
    parser.add_argument(
        '--save_results',
        action='store_true',
        default=True,
        help='Save results to text file (default: True)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress (default: True)'
    )
    
    args = parser.parse_args()
    
    stats = main(args)