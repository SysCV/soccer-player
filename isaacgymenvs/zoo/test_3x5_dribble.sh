env_name=("ramp" "low" "mid" "high" "gravel")
env_array=(
    "task.sim.gravity=[0.7,0.0,-9.5]" 
    "task.env.random_params.ball_drag.range_low=-0.1 task.env.random_params.ball_drag.range_high=0.1" 
    "task.env.random_params.ball_drag.range_low=0.1 task.env.random_params.ball_drag.range_high=0.3" 
    "task.env.random_params.ball_drag.range_low=0.3 task.env.random_params.ball_drag.range_high=0.5" 
    "task.env.terrain.type='trimesh'")

pt_name=(
    "baseline" 
"context" 
"PID")
pt_array=(
    "~task.env.priviledgeStates.dof_stiff ~task.env.priviledgeStates.dof_damp ~task.env.priviledgeStates.dof_calib ~task.env.priviledgeStates.payload ~task.env.priviledgeStates.com ~task.env.priviledgeStates.friction ~task.env.priviledgeStates.restitution ~task.env.priviledgeStates.ball_restitution ~task.env.priviledgeStates.ball_mass ~task.env.priviledgeStates.ball_states_v_1 ~task.env.priviledgeStates.ball_states_p_1 ~task.env.priviledgeStates.ball_states_v_2 ~task.env.priviledgeStates.ball_states_p_2 checkpoint='./checkpoints/dribble-baseline-106.pth'" 
    "checkpoint='./checkpoints/dribble-context-106.pth'" 
    "checkpoint='./checkpoints/dribble-PID-106.pth'"
)

# Get the length of the array
pt_length=${#pt_name[@]}
env_length=${#env_name[@]}

for i in $(seq 0 $((pt_length - 1))); do
    for j in $(seq 0 $((env_length - 1))); do
        python train.py task=Go1DribbleTest train=Go1DribblePPOsea test=true headless=true ${pt_array[$i]} ${env_array[$j]} task.env.log_env_name=${env_name[$j]} task.env.log_pt_name=${pt_name[$i]}
    done
done

