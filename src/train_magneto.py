#!/usr/bin/env python3
# %%
from magneto_env import MagnetoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from magneto_policy_learner import CustomActorCriticPolicy

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def train_ppo (env, path, rel_path, timesteps):
    # . Training    
    checkpoint_callback = CheckpointCallback(
        # save_freq=10,
        save_freq=100000,
        save_path=path + rel_path + 'weights/',
        name_prefix='magneto',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # - Start from scratch or load specified weights
    # model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    # model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    model = PPO.load(path + rel_path + 'breakpoint.zip', env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    print(model.policy)
    
    # # - Training
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback, progress_bar=True)
        
        for i in range(10):
            obs, _ = env.reset()
            over = False
            counter = 0
            while not over:
                action, _states = model.predict(obs)
                obs, rewards, over, _, _ = env.step(action)
                env.render()
                counter += 1
        env.close()
        
    finally:
        model.save(path + rel_path + 'breakpoint.zip')
        
def eval_ppo (env, path, rel_path, iterations):
    # . Evaluation
    model = PPO.load(path + rel_path + 'breakpoint.zip')
    # model = PPO.load(path + rel_path + 'weights/magneto_2800000_steps.zip')
    
    for _ in range(iterations):
        obs, _ = env.reset()
        over = False
        counter = 0
        while not over:
            action, _states = model.predict(obs)
            obs, rewards, over, _, _ = env.step(action)
            env.render()
            counter += 1
    env.close()

def test_ppo (env, path, rel_path):
    model = PPO.load(path + rel_path + 'breakpoint.zip')
    
    obs, _ = env.reset()
    for i in range(3):
        action, _states = model.predict(obs)
        print(f'Action: {action}')
        obs, rewards, over, _, _ = env.step(action)
        print(f'Reward: {rewards}')

def main ():
    # . Trying to learn SOMETHING
    path = '/home/steven/magneto_ws/outputs/'
    
    env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=0)
    rel_path = 'independent_walking/dqn/custom_mlp/projection/'
    # rel_path = 'independent_walking/ppo/scaled_obs_space_body/'
    
    # . Training
    # train_ppo(env, path, rel_path, 20000000)
    
    # . Evaluation
    eval_ppo(env, path, rel_path, 10)
    
    # . Test
    # test_ppo(env, path, rel_path)

if __name__ == "__main__":
    main()

# %%
