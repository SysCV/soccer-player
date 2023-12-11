env_name=("zero" "mid")
env_array=(
    "task.env.random_params.ball_drag.range_low=0.0 task.env.random_params.ball_drag.range_high=0.01" 
    "task.env.random_params.ball_drag.range_low=0.2 task.env.random_params.ball_drag.range_high=0.21" 
    )

pt_name=("baseline"
 "PID"
)
pt_array=(
    "~task.env.priviledgeStates.dof_stiff ~task.env.priviledgeStates.dof_damp ~task.env.priviledgeStates.dof_calib ~task.env.priviledgeStates.payload ~task.env.priviledgeStates.com ~task.env.priviledgeStates.friction ~task.env.priviledgeStates.restitution ~task.env.priviledgeStates.ball_mass ~task.env.priviledgeStates.ball_restitution ~task.env.priviledgeStates.ball_states_v_1 ~task.env.priviledgeStates.ball_states_p_1 ~task.env.priviledgeStates.ball_states_v_2 ~task.env.priviledgeStates.ball_states_p_2 checkpoint='./checkpoints/dribble-baseline-17.pth'" 
    "checkpoint='./checkpoints/dribble-PID-17.pth'"
)

# Get the length of the array
pt_length=${#pt_name[@]}
env_length=${#env_name[@]}

for i in $(seq 0 $((pt_length - 1))); do
    for j in $(seq 0 $((env_length - 1))); do
        python train.py task=Go1DribbleTraj train=Go1DribblePPOsea test=true headless=true ${pt_array[$i]} ${env_array[$j]} task.env.log_env_name=${env_name[$j]} task.env.log_pt_name=${pt_name[$i]}
    done
done

