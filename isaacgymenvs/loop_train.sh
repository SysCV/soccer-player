train_args=("Go1PPO" "Go1PPOrnn" "Go1PPOhistory" "Go1PPOsea" "Go1PPOrma"  "Go1PPOteacher" "Go1PPOwaq" "Go1PPOteacher")
# train_args=("Go1PPO" "Go1PPOrnn" "Go1PPOhistory" "Go1PPOsea" "Go1PPOrma"  "Go1PPOteacher" "Go1PPOwaq" "Go1PPOteacher")
seeds=(42 17 13) #) 2 17 13)

for seed in "${seeds[@]}"; do
    for train_arg in "${train_args[@]}"; do
        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg" seed="$seed" max_iterations=3000 wandb_group="8-30-ramdom-for-deploy"
    done
done



#     for seed in "${seeds[@]}"; do
#         # python train.py task=Go1 train="Go1PPOteacher" wandb_activate=true capture_video=true experiment="fix_Go1teacher" seed="$seed" max_iterations=3000

#         python train.py task=Go1 train="Go1PPOteacher" wandb_activate=true capture_video=true experiment="Go1teacher" seed="$seed" max_iterations=3000 task.env.randomCommandVelocityRanges.num_bins_yaw=11 task.env.randomCommandVelocityRanges.num_bins_x=11 task.env.wandb_extra_log=True
#     done
# done



# task.env.randomCommandVelocityRanges.num_bins_yaw = 11
# task.env.randomCommandVelocityRanges.num_bins_x = 11

# python train.py task=Go1 experiment="forward" seed="42" max_iterations=3000 wandb_activate=true capture_video=true task.env.randomCommandVelocityRanges.linear_x="[ -1.0, 3.0]" task.env.randomCommandVelocityRanges.linear_x_init="[ -1.0, 3.0]"
# wandb_activate=true capture_video=true
