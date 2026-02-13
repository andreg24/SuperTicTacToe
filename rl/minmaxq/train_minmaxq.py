"""
Main training script for MinMaxQ on Ultimate Tic Tac Toe.
Compatible with the base environment (no reward shaping).

python train_minmaxq.py \
    --episodes 100000 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.99995 \
    --lr 0.0001 \
    --batch_size 128 \
    --no_swap \
    --save_path weights_no_fixed_opponent \
    --plot \
    2>&1 | tee training_stdout_no_fixed_opponent.txt
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../")) 

from SuperTicTacToe.ultimatetictactoe import ultimatetictactoe

from agent import MinMaxQAgent
from train import train_minmaxq, evaluate_vs_random

from enhanced_train_metrics import (
    enhanced_train_minmaxq, 
    plot_enhanced_stats,
    evaluate_detailed
)


def plot_training_stats(stats: dict, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    ax = axes[0, 0]
    ax.plot(stats['episode_rewards_1'], label='Agent 1', alpha=0.6)
    ax.plot(stats['episode_rewards_2'], label='Agent 2', alpha=0.6)
    window = 50
    if len(stats['episode_rewards_1']) > window:
        avg1 = np.convolve(stats['episode_rewards_1'], np.ones(window)/window, mode='valid')
        avg2 = np.convolve(stats['episode_rewards_2'], np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(stats['episode_rewards_1'])), avg1, 'b-', linewidth=2, label='Agent 1 (avg)')
        ax.plot(range(window-1, len(stats['episode_rewards_2'])), avg2, 'orange', linewidth=2, label='Agent 2 (avg)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True)

    # Episode lengths
    ax = axes[0, 1]
    ax.plot(stats['episode_lengths'], alpha=0.6)
    if len(stats['episode_lengths']) > window:
        avg_len = np.convolve(stats['episode_lengths'], np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(stats['episode_lengths'])), avg_len, 'r-', linewidth=2, label='Average')
        ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Turns')
    ax.set_title('Episode Lengths')
    ax.grid(True)

    # Losses
    ax = axes[1, 0]
    if len(stats['losses_1']) > 0:
        ax.plot(stats['losses_1'], label='Agent 1', alpha=0.6)
        ax.plot(stats['losses_2'], label='Agent 2', alpha=0.6)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (MSE)')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')

    # Evaluation win rates
    ax = axes[1, 1]
    if len(stats['eval_win_rates']) > 0:
        episodes = stats['eval_episodes_list']
        win1 = [wr['agent1_wins'] for wr in stats['eval_win_rates']]
        win2 = [wr['agent2_wins'] for wr in stats['eval_win_rates']]
        draws = [wr['draws'] for wr in stats['eval_win_rates']]
        ax.plot(episodes, win1, 'g-o', label='Agent 1 wins')
        ax.plot(episodes, win2, 'r-o', label='Agent 2 wins')
        ax.plot(episodes, draws, 'b-o', label='Draws')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Percentage')
        ax.set_title('Evaluation Win Rates')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    env = ultimatetictactoe.env()
    print("Environment created.")

    agent1 = MinMaxQAgent(
        name="player_1",
        player_id=0,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        device=device,
        mode='train',
        use_double_dqn=args.double_dqn,
    )

    agent2 = MinMaxQAgent(
        name="player_2",
        player_id=1,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        device=device,
        mode='train',
        use_double_dqn=args.double_dqn,
    )

    print(f"Agents created. Q-network parameters: {sum(p.numel() for p in agent1.q_network.parameters())}")

    # stats = train_minmaxq(
    #     env=env,
    #     agent1=agent1,
    #     agent2=agent2,
    #     num_episodes=args.episodes,
    #     batch_size=args.batch_size,
    #     update_freq=args.update_freq,
    #     target_update_freq=args.target_update_freq,
    #     buffer_capacity=args.buffer_capacity,
    #     enable_swap=args.enable_swap,
    #     eval_freq=args.eval_freq,
    #     eval_episodes=args.eval_episodes,
    #     fixed_opponent=args.fixed_opponent,
    #     fixed_phase_episodes=args.fixed_phase_episodes,
    #     pool_size=args.pool_size,
    # )
    stats, tracker = enhanced_train_minmaxq(
        env=env,
        agent1=agent1,
        agent2=agent2,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        update_freq=args.update_freq,
        target_update_freq=args.target_update_freq,
        buffer_capacity=args.buffer_capacity,
        enable_swap=args.enable_swap,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
    )

    print("\nTraining complete!")

    eval_random = evaluate_vs_random(env, agent1, num_episodes=200)
    print(f"Agent1 vs Random:  Wins: {eval_random['agent_wins']:.1f}%,  "
          f"Losses: {eval_random['random_wins']:.1f}%,  Draws: {eval_random['draws']:.1f}%")

    if args.save_path:
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        agent1.save(save_dir / "agent1.pt")
        agent2.save(save_dir / "agent2.pt")
        print(f"\nModels saved to {save_dir}")

    # if args.plot:
    #     plot_path = Path(args.save_path) / "training_stats.png" if args.save_path else None
    #     plot_training_stats(stats, save_path=plot_path)
    if args.plot:
        plot_path = Path(args.save_path) / "training_stats_enhanced.png" if args.save_path else None
        plot_enhanced_stats(stats, save_path=plot_path)
        
        # Stampa summary delle metriche
        print("\n" + "="*70)
        print("METRICS SUMMARY")
        print("="*70)
        summary = tracker.get_summary()
        for key, values in summary.items():
            print(f"{key}:")
            print(f"  Mean: {values['mean']:.3f}")
            print(f"  Std:  {values['std']:.3f}")
            print(f"  Min:  {values['min']:.3f}")
            print(f"  Max:  {values['max']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MinMaxQ agents on Ultimate Tic Tac Toe")
    # Training
    parser.add_argument('--episodes', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--target_update_freq', type=int, default=2000)
    parser.add_argument('--buffer_capacity', type=int, default=100000)
    # Agent
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=float, default=0.99995)
    parser.add_argument('--double_dqn', action='store_true', default=True)
    parser.add_argument('--no_double_dqn', action='store_false', dest='double_dqn')
    # Options
    parser.add_argument('--enable_swap', action='store_true', default=False)  # Changed
    parser.add_argument('--swap', action='store_true', dest='enable_swap')  # Add explicit flag
    parser.add_argument('--no_swap', action='store_false', dest='enable_swap')
    parser.add_argument('--fixed_opponent', action='store_true', default=False, help='Train with fixed opponent (latest checkpoint)')
    parser.add_argument('--fixed_phase_episodes', type=int, default=5000, help='Episodes per fixed‑opponent phase')
    parser.add_argument('--pool_size', type=int, default=30, help='Number of past checkpoints to keep in pool')
    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--eval_episodes', type=int, default=20)
    # Saving & plotting
    parser.add_argument('--save_path', type=str, default='/weights')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    args = parser.parse_args()
    main(args)