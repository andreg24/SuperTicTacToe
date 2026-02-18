#!/usr/bin/env python3
"""
Train MinMaxQ with DENSE REWARDS (reward shaping).
Compare vs baseline sparse rewards.
"""

import argparse
import torch
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ultimatetictactoe import ultimatetictactoe
from agent import MinMaxQAgent
from enhanced_train_metrics import enhanced_train_minmaxq, plot_enhanced_stats
from train import evaluate_vs_random


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================================
    # ENVIRONMENT WITH DENSE REWARDS
    # ============================================================
    env = ultimatetictactoe.env(dense_rewards=args.dense_rewards)
    
    reward_type = "DENSE" if args.dense_rewards else "SPARSE"
    print(f"Environment created with {reward_type} rewards")
    
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
        use_double_dqn=True,
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
        use_double_dqn=True,
    )
    
    print(f"Agents created. Parameters: {sum(p.numel() for p in agent1.q_network.parameters())}")
    
    # ============================================================
    # TRAINING
    # ============================================================
    print(f"\nStarting training with {reward_type} rewards...")
    
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
    
    # ============================================================
    # FINAL EVALUATION
    # ============================================================
    eval_random = evaluate_vs_random(env, agent1, num_episodes=200)
    print(f"\nFINAL EVALUATION ({reward_type} rewards):")
    print(f"  Agent vs Random:")
    print(f"    Wins:   {eval_random['agent_wins']:.1f}%")
    print(f"    Losses: {eval_random['random_wins']:.1f}%")
    print(f"    Draws:  {eval_random['draws']:.1f}%")
    
    # ============================================================
    # SAVE MODELS
    # ============================================================
    if args.save_path:
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        agent1.save(save_dir / "agent1.pt")
        agent2.save(save_dir / "agent2.pt")
        print(f"\nModels saved to {save_dir}")
    
    # ============================================================
    # PLOT RESULTS
    # ============================================================
    if args.plot:
        plot_path = Path(args.save_path) / "training_stats_enhanced.png" if args.save_path else None
        plot_enhanced_stats(stats, save_path=plot_path)
        
        # Print summary
        print("\n" + "="*70)
        print(f"METRICS SUMMARY ({reward_type} rewards)")
        print("="*70)
        summary = tracker.get_summary()
        for key, values in summary.items():
            print(f"{key}:")
            print(f"  Mean: {values['mean']:.3f}")
            print(f"  Std:  {values['std']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MinMaxQ with Dense/Sparse Rewards")
    
    # Reward type
    parser.add_argument('--dense_rewards', action='store_true', default=False,
                        help='Use dense rewards (+0.1 for sub-board wins)')
    
    # Training
    parser.add_argument('--episodes', type=int, default=60000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--target_update_freq', type=int, default=1000)
    parser.add_argument('--buffer_capacity', type=int, default=100000)
    
    # Agent
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=float, default=0.99995)
    
    # Options
    parser.add_argument('--enable_swap', action='store_true', default=False)
    parser.add_argument('--no_swap', action='store_false', dest='enable_swap')
    
    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--eval_episodes', type=int, default=20)
    
    # Saving & plotting
    parser.add_argument('--save_path', type=str, default='weights_dense_rewards')
    parser.add_argument('--plot', action='store_true', default=True)
    
    args = parser.parse_args()
    main(args)