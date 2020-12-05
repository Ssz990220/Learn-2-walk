import gym, ray
from ray.rllib.agents import ppo
from ray import tune
from pybulletgym.envs.roboschool.envs.locomotion.humanoid_env import HumanoidBulletEnv
from ray.tune.registry import register_env

def env_creator(env_config):
    return HumanoidBulletEnv()

register_env('HumanoidPyBullet-v0',env_creator)

ray.init()
tune.run(
    "PPO",
    stop={'episode_reward_mean':6000},
    config={"env":'HumanoidPyBullet-v0',
            'framework':'tf',
            'num_workers':10,
            'gamma':0.995,
            'lambda':0.95,
            'clip_param':0.2,
            'kl_coeff':1.0,
            'num_sgd_iter':20,
            'lr':0.001,
            'sgd_minibatch_size':32768,
            'horizon':5000,
            'train_batch_size':320000,
            'model':{'free_log_std':True},
            'batch_mode':'complete_episodes',
            'observation_filter':'MeanStdFilter',
            'num_gpus':0}
)
