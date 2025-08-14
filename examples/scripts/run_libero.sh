#!/bin/bash
proj_name=DSRL_pi0_Libero
device_id=4

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=4
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

pip install mujoco==3.3.1

python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--prefix dsrl_pi0_libero \
--suffix collect_traj_for_FT \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 1 \
--multi_grad_step 20 \
--start_online_updates 10 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 20 \
--hidden_dims 128 \