#!/usr/bin/env python3
# %%
from magneto_env import MagnetoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from magneto_policy_learner import CustomActorCriticPolicy

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

def main ():
    path = '/home/steven/magneto_ws/outputs/'
    env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=5)
    rel_path = 'bandwidth/'
    
    # . Evaluation
    eval_ppo(env, path, rel_path, 5)

if __name__ == "__main__":
    main()

# %%
