train_args=("Go1PPO")
# train_args=("Go1PPO" "Go1PPOrnn" "Go1PPOhistory" "Go1PPOsea" "Go1PPOrma"  "Go1PPOteacher" "Go1PPOwaq" "Go1PPOteacher")
seeds=(42) #) 2 17 13)

for seed in "${seeds[@]}"; do
    for train_arg in "${train_args[@]}"; do
        # === what if frequence is 1.0 2.0 3.0, walk and tort===
        # gait_condition:
        # trotting [0.5,0,0] pacing [0,0,0.5] walking [0.0,0.25,0.5]
        # phases: 0.5
        # offsets: 0.0
        # bounds: 0.0
        # kappa: 0.07
        # duration: 0.9 # for stance phase
        # frequency: 3.0
        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg f3 walk" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.gait_condition.frequency=3.0 task.env.gait_condition.phases=0.0 task.env.gait_condition.offsets=0.25 task.env.gait_condition.bounds=0.5

        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg f2" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.gait_condition.frequency=2.0

        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg f1" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.gait_condition.frequency=1.0



        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg f1 walk" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.gait_condition.frequency=1.0 task.env.gait_condition.phases=0.0 task.env.gait_condition.offsets=0.25 task.env.gait_condition.bounds=0.5

        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg f2 walk" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.gait_condition.frequency=2.0 task.env.gait_condition.phases=0.0 task.env.gait_condition.offsets=0.25 task.env.gait_condition.bounds=0.5

        # python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg 0.9 s" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.rewards.rewardScales.ground_impact_posi=0.2 task.env.rewards.rewardScales.ground_speed_posi=0.2 task.env.gait_condition.duration=0.9

        # what if kappa lower, difference between force and speed

        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg 0 0" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.rewards.rewardScales.ground_impact_posi=0.0 task.env.rewards.rewardScales.ground_speed_posi=0.0 task.env.gait_condition.duration=0.5 task.env.gait_condition.kappa=0.2

        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg 1 0" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.rewards.rewardScales.ground_impact_posi=0.1 task.env.rewards.rewardScales.ground_speed_posi=0.0 task.env.gait_condition.duration=0.5 task.env.gait_condition.kappa=0.2


        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg 4 0" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.rewards.rewardScales.ground_impact_posi=0.4 task.env.rewards.rewardScales.ground_speed_posi=0.0 task.env.gait_condition.duration=0.5 task.env.gait_condition.kappa=0.2

        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg 0 1" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.rewards.rewardScales.ground_impact_posi=0.0 task.env.rewards.rewardScales.ground_speed_posi=0.1 task.env.gait_condition.duration=0.5 task.env.gait_condition.kappa=0.2

        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg 0 4" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.rewards.rewardScales.ground_impact_posi=0.0 task.env.rewards.rewardScales.ground_speed_posi=0.4 task.env.gait_condition.duration=0.5 task.env.gait_condition.kappa=0.2

        python train.py task=Go1 train="$train_arg" wandb_activate=true capture_video=true experiment="$train_arg 2 2" seed="$seed" max_iterations=3000 wandb_group="9-19-gentle" max_iterations=3000 task.env.rewards.rewardScales.ground_impact_posi=0.2 task.env.rewards.rewardScales.ground_speed_posi=0.2 task.env.gait_condition.duration=0.5 task.env.gait_condition.kappa=0.2


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
