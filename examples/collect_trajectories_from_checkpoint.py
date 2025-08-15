from flax.training import checkpoints
from train_utils_sim import perform_control_eval
import argparse
from train_utils_sim import _get_libero_env
from libero.libero import benchmark
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from train_sim import DummyEnv
from jaxrl2.utils.general_utils import add_batch_dim
import tensorflow as tf
from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download
import os
import tempfile
from jaxrl2.data import ReplayBuffer

def main(variant):
    print("VARIANT IS: ", variant)
    import ipdb; ipdb.set_trace()
    assert variant.env == 'libero'

    ## boilerplate setup copied from train_sim.py

    ## seting up args, dirs, and naming
    tf.config.set_visible_devices([], "GPU")
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
    
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}" + f"_{variant.checkpoint_dir.replace('/', '_')}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.checkpoint_dir.replace('/', '_')}"

    print("EXPNAME IS: ", expname)
   
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)

    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_90"]()
    task_id = 57 ## (YY): hard-coding since we only checking on cream cheese task
    task = task_suite.get_task(task_id)
    eval_env, task_description = _get_libero_env(task, 256, variant.seed)
    variant.task_description = task_description
    variant.env_max_reward = 1
    variant.max_timesteps = 400

    ## TODO(YY): edit this later if want to add wandb logging
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)


    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    # initialize pi0 policy agent will query
    config = openpi_config.get_config("pi0_libero")
    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)

    
    # initialize and restore our agent
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)
    agent.restore_checkpoint(variant.checkpoint_dir)

    ## replay buffer setup
    online_buffer_size = variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)

    # i can be anything except 0, perform_control_eval initializs random normal noise if i ==0, otherwise uses the SAC agent to generate noise if i > 0
    # other than that, i just used in some wandb logging
    print(f"Will perform {variant.eval_episodes} evals, so expect that many trajectories to be saved to {variant.outputdir} !! ")
    perform_control_eval(agent, eval_env, 1, variant, wandb_logger, agent_dp)


# if __name__ == "main":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--checkpoint_dir", type=str, required=True, help="The directory to load the checkpoint from."
#     )
#     parser.add_argument(
#         "--prefix", type=str, required=True, help="The prefix the saved checkpoints used, needed to locate the correct checkpoint."
#     )
#     parser.add_argument(
#         "--step", type=int, required=True, help="Which train step in the checkpoint_dir to load"."
#     )
#     parser.add_argument(
#         "--outdir", type=str, required=True, help="The directory to save the trajectories to."
#     )    
#     parser.add_argument(
#         "--seed", default=0, type=int, required=False, help="The seed to use for the environment."
#     )
#     parser.add_argument(
#         "--resize_image", default=256, type=int, required=False, help="The resize image size to use for the environment."
#     )
#     parser.add_argument(
#         "--num_cameras", default=1, type=int, required=False, help="The number of cameras to use for the environment."
#     )
#     parser.add_argument(
#         "--add_states", default=False, type=bool, required=False, help="Whether to add states to the observation."
#     )
#     parser.add_argument(
#         "--env", default="libero", type=str, required=False, help="The environment to use."
#     )
#     args = parser.parse_args()
#     main(args)