import gym, ray, time
import pybulletgym
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from pybulletgym.envs.roboschool.envs.locomotion.humanoid_env import HumanoidBulletEnv

def env_creator(env_config):
    return HumanoidBulletEnv()

register_env('HumanoidPyBullet-v0',env_creator)
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
checkpoint_path = './checkpoint_197'
ray.init(include_dashboard=False,object_store_memory=6000000000)
agent = ppo.PPOTrainer(config=config, env='HumanoidPyBullet-v0')
agent.restore(checkpoint_path)

env = gym.make('HumanoidPyBulletEnv-v0')
env.render()
ob = env.reset()
reward = 0

while True:
    action, _states = agent.compute_action(ob)
    ob, r, done, info = env.step(action)
    reward += r
    time.sleep(0.01)
    if done:
        ob = env.reset()
        print('r is {}'.format(r))
        print('Episode reward is {}'.format(reward))
        reward = 0