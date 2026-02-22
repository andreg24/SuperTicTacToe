$ErrorActionPreference = "Stop"

# Use venv Python directly (recommended)
$python = "..\venv\Scripts\python.exe"

# # -----------------
# # basic paper
# # -----------------
# & $python reinforce_training.py `
#   --num_episodes 18000 `
#   --enable_swap `
#   --enable_transform `
#   --px 0 `
#   --pt 0 `
#   --checkpoint_rate 100 `
#   --validation_rate 100 `
#   --experiment_name long_run `
#   --eps 0.3 `
#   --a1_learning_power 6 `
#   --a1_learning_const 1.0 `
#   --a1_exploration_power 2.0 `
#   --a1_exploration_const 1.0 `
#   --a2_learning_power 10.5 `
#   --a2_learning_const 1.0 `
#   --a2_exploration_power 1.0 `
#   --a2_exploration_const 1.0 `
#   --weights_path "" `
#   --weights_name "" `
#   --device cpu

# # -----------------
# # paper with const
# # -----------------
# & $python reinforce_training.py `
#   --num_episodes 14000 `
#   --enable_swap `
#   --enable_transform `
#   --px 0 `
#   --pt 0 `
#   --checkpoint_rate 100 `
#   --validation_rate 100 `
#   --experiment_name long_run `
#   --eps 0.3 `
#   --a1_learning_power 6 `
#   --a1_learning_const 1.0 `
#   --a1_exploration_power 2.0 `
#   --a1_exploration_const 1.7 `
#   --a2_learning_power 10.5 `
#   --a2_learning_const 10.0 `
#   --a2_exploration_power 1.0 `
#   --a2_exploration_const 0.75 `
#   --weights_path "rl/independent_algo/logs/checkpoints/long_run/2026_02_20_10_44_59" `
#   --weights_name "model_9000.pt" `
#   --device cpu



# # -----------------
# # slow high
# # -----------------
# & $python reinforce_training.py `
#   --num_episodes 9000 `
#   --enable_swap `
#   --enable_transform `
#   --px 0 `
#   --pt 0 `
#   --checkpoint_rate 100 `
#   --validation_rate 100 `
#   --experiment_name base_long_run_slow `
#   --eps 0.3 `
#   --a1_learning_power 0.0 `
#   --a1_learning_const 1e-6 `
#   --a1_exploration_power 1.0 `
#   --a1_exploration_const 1.0 `
#   --a2_learning_power 0.0 `
#   --a2_learning_const 1e-6 `
#   --a2_exploration_power 1.0 `
#   --a2_exploration_const 1.0 `
#   --weights_path "rl/independent_algo/logs/checkpoints/base_long_run_slow/2026_02_21_11_02_13" `
#   --weights_name "model_8000.pt" `
#   --device cpu


# -----------------
# fast high
# -----------------
& $python reinforce_training.py `
  --num_episodes 4100 `
  --enable_swap `
  --enable_transform `
  --px 0 `
  --pt 0 `
  --checkpoint_rate 100 `
  --validation_rate 100 `
  --experiment_name fast_l_high_e `
  --eps 0.3 `
  --a1_learning_power 0.0 `
  --a1_learning_const 1e-3 `
  --a1_exploration_power 0.0 `
  --a1_exploration_const 0.3 `
  --a2_learning_power 0.0 `
  --a2_learning_const 1e-3 `
  --a2_exploration_power 0.0 `
  --a2_exploration_const 0.3 `
  --weights_path "rl/independent_algo/logs/checkpoints/fast_l_high_e/2026_02_22_15_08_30" `
  --weights_name "model_36100.pt" `
  --device cpu

# # -----------------
# # slow low
# # -----------------
# & $python reinforce_training.py `
#   --num_episodes 9001 `
#   --enable_swap `
#   --enable_transform `
#   --px 0 `
#   --pt 0 `
#   --checkpoint_rate 100 `
#   --validation_rate 100 `
#   --experiment_name slow_l_low_e `
#   --eps 0.3 `
#   --a1_learning_power 0.0 `
#   --a1_learning_const 1e-6 `
#   --a1_exploration_power 0.0 `
#   --a1_exploration_const 0.1 `
#   --a2_learning_power 0.0 `
#   --a2_learning_const 1e-6 `
#   --a2_exploration_power 0.0 `
#   --a2_exploration_const 0.1 `
#   --device cpu