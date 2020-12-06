import gym
import pybulletgym
import time

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',type=str)
    parser.add_argument('--alg',type=str)
    parser.add_argument('--file', type=str)
    args = parser.parse_args()

    if args.env == 'w' or args.env == 'Walker2DPyBulletEnv-v0': 
        env = gym.make('Walker2DPyBulletEnv-v0')
    elif args.env == 'h' or args.env == 'HumanoidPyBulletEnv-v0':
        env = gym.make('HumanoidPyBulletEnv-v0')
    else:
        raise ValueError('No such environment!')    
    env.render()

    if args.alg == 'td3' or args.alg=='TD3':
        from stable_baselines import TD3
        model = TD3.load(args.file)
    elif args.alg == 'ddpg' or args.alg == 'DDPG':
        from stable_baselines import DDPG
        model = DDPG.load(args.file)
    elif args.alg == 'SAC' or args.alg == 'sac':
        from stable_baselines import SAC
        model = SAC.load(args.file)
    elif args.alg == 'ppo1' or args.alg == 'PPO1':
        from stable_baselines import PPO1
        model = PPO1.load(args.file)
    elif args.alg == 'ppo2' or args.alg == 'PPO2':
        from stable_baselines import PPO2
        model = PPO2.load(args.file)
    else:
        raise ValueError('No such algorithm')
    
    
    ob = env.reset()
    reward = 0

    while True:
        action, _states = model.predict(ob)
        ob, r, done, info = env.step(action)
        reward += r
        time.sleep(0.01)
        if done:
            ob = env.reset()
            print('r is {}'.format(r))
            print('Episode reward is {}'.format(reward))
            reward = 0