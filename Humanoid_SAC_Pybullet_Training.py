import gym
import pybulletgym
import numpy as np
import json
from pathlib import Path
from datetime import date, datetime
# import wandb
import os

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC, logger
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor

from stable_baselines_utils import *

def train(env_name, num_time_steps, policy_kwargs, eval_ep, eval_freq, ckpt_freq, load_model=None):
    env = gym.make(env_name)
    # env.render()
    env_ = gym.make(env_name)

    today = date.today()
    today = str(today).replace('-','_')
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    model_name = env_name + '_SAC_' + today + current_time
    Path('./run/'+model_name).mkdir(parents=True, exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), './run/' + model_name)
    env = Monitor(env, filename=path)
    ############################
    #          Logging         #
    ############################
    logger.configure(path)
    config = {}
    config['load']=[{'load_model':load_model}]
    config['eval']=[{'eval_freq':eval_freq, 'eval_ep':eval_ep}]
    config['ckpt']=[{'ckpt_freq':ckpt_freq}]
    config['policy']=[{'policy_network':policy_kwargs}]
    with open('./run/' + model_name + '/' + model_name + '.txt', 'w+') as outfile:
        json.dump(config, outfile, indent=4)

    ############################
    #         callback         #
    ############################
    callbacklist = []
    ckpt_callback = CheckpointCallback(save_freq=ckpt_freq, save_path='./run/' + model_name + '/ckpt', name_prefix='')
    eval_callback = EvalCallback_wandb_SAC(env_, n_eval_episodes=eval_ep, eval_freq=eval_freq, log_path=path)
    callbacklist.append(ckpt_callback)
    callbacklist.append(eval_callback)
    callback = CallbackList(callbacklist)

    ############################
    #            run           #
    ############################
    # policy_kwargs = dict(net_arch=[128, dict(vf=[256], pi=[16])])
    model = SAC(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=int(num_time_steps), log_interval=20, callback=callback)
    model.save(path+"SAC_Humanoid")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env',type=str, default='HumanoidPyBulletEnv-v0')
    parser.add_argument('--load_model',type=str, default=None)
    parser.add_argument('--total_timesteps', type=float, default=2e6)
    parser.add_argument('--eval_freq', type=int, default=20000)
    parser.add_argument('--eval_ep', type=int, default=10)
    parser.add_argument('--ckpt_freq', type=int, default=10000)
    parser.add_argument('--policy',type=dict, default={'net_arch':[128,64]})
    args = parser.parse_args()
    # wandb.init(project='Two-Finger-Pinching')

    train(env_name=args.env, num_time_steps=args.total_timesteps, policy_kwargs=None,
            eval_ep=args.eval_ep, eval_freq=args.eval_freq, ckpt_freq=args.ckpt_freq)
