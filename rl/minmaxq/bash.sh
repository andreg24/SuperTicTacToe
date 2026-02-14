#!/bin/bash

# ============================================================
# MinMaxQ Training Suite - NON-BLOCKING VERSION
# ============================================================

set -e  # Exit on error
cd ~/Downloads/Reinforcement/SuperTicTacToe/rl/minmaxq

# Disable interactive matplotlib backend
export MPLBACKEND=Agg

echo "============================================================"
echo "Starting Training Suite"
echo "Time: $(date)"
echo "============================================================"
echo ""

# ============================================================
# EXPERIMENT 1: No Fixed Opponent
# ============================================================
echo "============================================================"
echo "EXPERIMENT 1/7: No Fixed Opponent"
echo "Started: $(date)"
echo "============================================================"

python train_minmaxq.py \
    --episodes 100000 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.99995 \
    --lr 0.0001 \
    --batch_size 128 \
    --no_swap \
    --save_path ./weights_no_fixed_opponent \
    2>&1 | tee training_stdout_no_fixed_opponent.txt

echo "Experiment 1 completed: $(date)"
echo ""

# ============================================================
# EXPERIMENT 2: Slower Epsilon
# ============================================================
echo "============================================================"
echo "EXPERIMENT 2/7: Slower Epsilon"
echo "Started: $(date)"
echo "============================================================"

python train_minmaxq.py \
    --episodes 60000 \
    --epsilon_start 1.0 \
    --epsilon_end 0.02 \
    --epsilon_decay 0.99998 \
    --lr 0.0001 \
    --batch_size 128 \
    --fixed_opponent \
    --fixed_phase_episodes 5000 \
    --target_update_freq 2000 \
    --eval_freq 2000 \
    --save_path ./weights_slower_epsilon \
    2>&1 | tee stdout_slower_epsilon.txt

echo "Experiment 2 completed: $(date)"
echo ""

# ============================================================
# EXPERIMENT 3: Higher Learning Rate
# ============================================================
echo "============================================================"
echo "EXPERIMENT 3/7: Higher Learning Rate"
echo "Started: $(date)"
echo "============================================================"

python train_minmaxq.py \
    --episodes 60000 \
    --epsilon_start 1.0 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.99995 \
    --lr 0.0003 \
    --batch_size 128 \
    --fixed_opponent \
    --fixed_phase_episodes 5000 \
    --target_update_freq 2000 \
    --eval_freq 2000 \
    --save_path ./weights_higher_lr \
    2>&1 | tee stdout_higher_lr.txt

echo "Experiment 3 completed: $(date)"
echo ""

# ============================================================
# EXPERIMENT 4: Shorter Phase
# ============================================================
echo "============================================================"
echo "EXPERIMENT 4/7: Shorter Phase"
echo "Started: $(date)"
echo "============================================================"

python train_minmaxq.py \
    --episodes 60000 \
    --epsilon_start 1.0 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.99995 \
    --lr 0.0001 \
    --batch_size 128 \
    --fixed_opponent \
    --fixed_phase_episodes 3000 \
    --target_update_freq 2000 \
    --eval_freq 2000 \
    --save_path ./weights_shorter_phase \
    2>&1 | tee stdout_shorter_phase.txt

echo "Experiment 4 completed: $(date)"
echo ""

# ============================================================
# EXPERIMENT 5: Frequent Target Updates
# ============================================================
echo "============================================================"
echo "EXPERIMENT 5/7: Frequent Target Updates"
echo "Started: $(date)"
echo "============================================================"

python train_minmaxq.py \
    --episodes 60000 \
    --epsilon_start 1.0 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.99995 \
    --lr 0.0001 \
    --batch_size 128 \
    --fixed_opponent \
    --fixed_phase_episodes 5000 \
    --target_update_freq 1000 \
    --eval_freq 2000 \
    --save_path ./weights_frequent_target \
    2>&1 | tee stdout_frequent_target.txt

echo "Experiment 5 completed: $(date)"
echo ""

# ============================================================
# EXPERIMENT 6: Large Batch
# ============================================================
echo "============================================================"
echo "EXPERIMENT 6/7: Large Batch"
echo "Started: $(date)"
echo "============================================================"

python train_minmaxq.py \
    --episodes 60000 \
    --epsilon_start 1.0 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.99995 \
    --lr 0.0001 \
    --batch_size 256 \
    --fixed_opponent \
    --fixed_phase_episodes 5000 \
    --target_update_freq 2000 \
    --eval_freq 2000 \
    --save_path ./weights_large_batch \
    2>&1 | tee stdout_large_batch.txt

echo "Experiment 6 completed: $(date)"
echo ""

# ============================================================
# EXPERIMENT 7: COMBO Optimized
# ============================================================
echo "============================================================"
echo "EXPERIMENT 7/7: COMBO Optimized"
echo "Started: $(date)"
echo "============================================================"

python train_minmaxq.py \
    --episodes 50000 \
    --epsilon_start 1.0 \
    --epsilon_end 0.02 \
    --epsilon_decay 0.99998 \
    --lr 0.00025 \
    --batch_size 256 \
    --fixed_opponent \
    --fixed_phase_episodes 3000 \
    --target_update_freq 1000 \
    --eval_freq 1000 \
    --save_path ./weights_combo_optimized \
    2>&1 | tee stdout_combo_optimized.txt

echo "Experiment 7 completed: $(date)"
echo ""

# ============================================================
# ALL DONE
# ============================================================
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Results saved in:"
echo "  - weights_*/"
echo "  - *.txt (logs)"
echo ""
echo "Next: Check results and push to git"