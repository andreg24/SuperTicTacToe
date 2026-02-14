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
    --save_path weights_slower_epsilon \
    --plot \
    2>&1 | tee stdout_slower_epsilon.txt

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
    --save_path weights_higher_lr \
    --plot \
    2>&1 | tee stdout_higher_lr.txt

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
    --save_path weights_shorter_phase \
    --plot \
    2>&1 | tee stdout_shorter_phase.txt

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
    --save_path weights_frequent_target \
    --plot \
    2>&1 | tee stdout_frequent_target.txt

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
    --save_path weights_large_batch \
    --plot \
    2>&1 | tee stdout_large_batch.txt

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
    --save_path weights_combo_optimized \
    --plot \
    2>&1 | tee stdout_combo_optimized.txt

git add .
git commit -m "benchmark trainings"
git push origin main

