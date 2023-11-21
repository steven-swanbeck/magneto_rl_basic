#!/usr/bin/env python3
# %%
from magneto_env import MagnetoEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from magneto_policy_learner import CustomActorCriticPolicy

def eval_ppo (env, path, rel_path, iterations):
    # . Evaluation
    model = DQN.load(path + rel_path + 'breakpoint.zip')
    # model = DQN.load(path + rel_path + 'weights/magneto_675000_steps.zip')
    
    for _ in range(iterations):
        obs, _ = env.reset()
        # print(obs)
        # input('...')
        over = False
        counter = 0
        while not over:
            action, _states = model.predict(obs)
            obs, rewards, over, _, _ = env.step(action)
            env.render()
            # input("Next step...")
            counter += 1
    env.close()

def main ():
    path = '/home/steven/magneto_ws/outputs/'
    env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=10)
    # rel_path = 'dqn/leader_follower/multi_input/paraboloid_penalty/'
    rel_path = 'dqn/independent/multi_input/paraboloid_penalty/'
    
    # . Evaluation
    eval_ppo(env, path, rel_path, 5)

if __name__ == "__main__":
    main()

# %%
