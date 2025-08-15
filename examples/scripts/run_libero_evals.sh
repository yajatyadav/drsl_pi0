#!/bin/bash
# Job name:
#SBATCH --job-name=eval_and_collect_traj_for_pi0_FT
#SBATCH --export=ALL 
#
# Account:
#SBATCH --account=co_rail
#
# Partition:
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#Number of GPUs, this should generally be in the form "gpu:A5000:[1-4] with the type included
#SBATCH --gres=gpu:A40:4
#
# Number of nodes:
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.out   # Stdout file (%x=job name, %j=job ID)
#SBATCH --error=logs/%x_%j.err    # Stderr file
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs (GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu and A5000 in savio4_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=4
#

#
# Wall clock limit:
#SBATCH --time=24:00:00


proj_name=DSRL_pi0_Libero_EVALS
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

pip install mujoco==3.3.1

python3 examples/launch_eval_sim.py \
# --algorithm pixel_sac \
--env libero \
--prefix dsrl_pi0_libero_evals \
--suffix eval_and_collect_traj_for_pi0_FT__using_ckpt_ \
--wandb_project ${proj_name} \
# --batch_size 256 \
# --discount 0.999 \
--seed 0 \
# --max_steps 500000  \
# --eval_interval 10000 \
# --log_interval 500 \
--eval_episodes 10 \ # this is controlling the number of trajectories to collect!!
# --multi_grad_step 20 \
# --start_online_updates 500 \
--resize_image 64 \
# --action_magnitude 1.0 \
--query_freq 20 \
# --hidden_dims 128 \
# --checkpoint_interval 10000 \
--checkpoint_dir "INVALID_PATH_INTETIONALLY!!" \