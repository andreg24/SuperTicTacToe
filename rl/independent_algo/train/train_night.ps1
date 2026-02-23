$ErrorActionPreference = "Stop"

# Use venv Python directly (recommended)
$python = "..\venv\Scripts\python.exe"


# # -----------------
# # fast high
# # -----------------
# & $python reinforce_training.py `
#   --num_episodes 40000 `
#   --enable_swap `
#   --enable_transform `
#   --px 0 `
#   --pt 0 `
#   --checkpoint_rate 100 `
#   --validation_rate 100 `
#   --experiment_name refined_mixed_const `
#   --eps 0.3 `
#   --a1_learning_power 6.0 `
#   --a1_learning_const 2 `
#   --a1_exploration_power 2.0 `
#   --a1_exploration_const 1.2 `
#   --a2_learning_power 10.5 `
#   --a2_learning_const 50 `
#   --a2_exploration_power 1.0 `
#   --a2_exploration_const 0.75 `
#   --device cpu

# -----------------
# slow low
# -----------------
& $python reinforce_training.py `
  --num_episodes 9001 `
  --enable_swap `
  --enable_transform `
  --px 0 `
  --pt 0 `
  --checkpoint_rate 100 `
  --validation_rate 100 `
  --experiment_name slow_l_low_e `
  --eps 0.3 `
  --a1_learning_power 0.0 `
  --a1_learning_const 1e-6 `
  --a1_exploration_power 0.0 `
  --a1_exploration_const 0.1 `
  --a2_learning_power 0.0 `
  --a2_learning_const 1e-6 `
  --a2_exploration_power 0.0 `
  --a2_exploration_const 0.1 `
  --weights_path "rl/independent_algo/logs/checkpoints/slow_l_low_e/2026_02_21_22_09_08" `
  --weights_name "model_9000.pt" `
  --device cpu
