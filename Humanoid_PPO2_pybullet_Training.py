import gym
import pybulletgym
from stable_baselines.common.policies import MlpPolicy
import wandb
import json
import os
from pathlib import Path
from datetime import date, time, datetime
from stable_baselines import logger, PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines_utils import *

def make_env(env_id, rank, seed=0, path=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env = Monitor(env, filename=path)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def train(env_name, num_time_steps, policy_kwargs, eval_ep, eval_freq, ckpt_freq, load_model=None):
    today = date.today()
    today = str(today).replace('-','_')
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    model_name = env_name + '_PPO2_' + today + current_time
    Path('./run/'+model_name).mkdir(parents=True, exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), './run/' + model_name)
    num_cpu = 24
    env = SubprocVecEnv([make_env(env_name, i, path=path) for i in range(num_cpu)])
    env_ = gym.make(env_name)


    ############################
    #         callback         #
    ############################   
    callbacklist = []
    eval_callback = EvalCallback_wandb(env_, n_eval_episodes=eval_ep, eval_freq=eval_freq, log_path=path)
    ckpt_callback = CheckpointCallback(save_freq=ckpt_freq, save_path='./run/' + model_name + '/ckpt', name_prefix='')
    callbacklist.append(eval_callback)
    callbacklist.append(ckpt_callback)
    callback = CallbackList(callbacklist)

    if load_model:
        model = PPO2.load(env=env, load_path=load_model)
    else:
        model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs)

    ############################
    #          Logging         #
    ############################
    logger.configure()
    config = {}
    config['load']=[{'load_model':load_model}]
    config['eval']=[{'eval_freq':eval_freq, 'eval_ep':eval_ep}]
    config['ckpt']=[{'ckpt_freq':ckpt_freq}]
    config['policy']=[{'policy_network':policy_kwargs}]
    with open('./run/' + model_name + '/' + model_name + '.txt', 'w+') as outfile:
        json.dump(config, outfile, indent=4)
    ############################
    #            run           #
    ############################
   
    model.learn(total_timesteps=int(num_time_steps), callback=callback)
    model.save(path+'/finish')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env',type=str, default='HumanoidPyBulletEnv-v0')
    parser.add_argument('--load_model',type=str, default=None)
    parser.add_argument('--n', type=float, default=2e8)
    parser.add_argument('--eval_freq', type=int, default=20000)
    parser.add_argument('--eval_ep', type=int, default=20)
    parser.add_argument('--ckpt_freq', type=int, default=5000)
    parser.add_argument('--policy',type=dict, default={'net_arch':[128,64]})
    args = parser.parse_args()
    #if rank == 0:
    #   wandb.init(project='Big_Data_Project')
    # print(args.load_model)

    train(env_name=args.env, num_time_steps=args.n, policy_kwargs=None,
            eval_ep=args.eval_ep, eval_freq=args.eval_freq, ckpt_freq=args.ckpt_freq)
            # load_model=str(args.load_model))
