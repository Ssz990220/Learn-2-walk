import gym
import pybulletgym
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
import wandb
import json
import os
from pathlib import Path
from datetime import date, time, datetime
from mpi4py import MPI
from stable_baselines import logger
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines_utils import *

def train(env_name, num_time_steps, policy_kwargs, eval_ep, eval_freq, ckpt_freq, load_model=None):
    env=gym.make(env_name)
    env_ = gym.make(env_name)
    rank = MPI.COMM_WORLD.Get_rank()
    today = date.today()
    today = str(today).replace('-','_')
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    model_name = env_name + '_PPO1_' + today + current_time
    Path('./run/'+model_name).mkdir(parents=True, exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), './run/' + model_name)


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
        model = PPO1.load(env=env, load_path=load_model)
    else:
        model = PPO1(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs)

    ############################
    #          Logging         #
    ############################
    if rank==0:
        logger.configure()
        config = {}
        config['load']=[{'load_model':load_model}]
        config['eval']=[{'eval_freq':eval_freq, 'eval_ep':eval_ep}]
        config['ckpt']=[{'ckpt_freq':ckpt_freq}]
        config['policy']=[{'policy_network':policy_kwargs}]
        with open('./run/' + model_name + '/' + model_name + '.txt', 'w+') as outfile:
            json.dump(config, outfile, indent=4)
    else:
        logger.configure(format_strs=[])
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #if rank == 0:
    #   wandb.init(project='Big_Data_Project')
    # print(args.load_model)

    train(env_name=args.env, num_time_steps=args.n, policy_kwargs=None,
            eval_ep=args.eval_ep, eval_freq=args.eval_freq, ckpt_freq=args.ckpt_freq)
            # load_model=str(args.load_model))
