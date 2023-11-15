# %%
import numpy as np

test = np.zeros((20, 1, 10), dtype=np.float32)

# %%
new_state = np.random.random((1, 10))

# %%
reshaped = np.reshape(new_state, (1, 1, 10))

# %%
test2 = np.vstack((test, reshaped))[1:]

# %%
import torch

# %%
n1 = np.array([1, 2, 3])
n2 = np.array([1, 2, 3])
print(n1 / n2)
# %%
test = np.array([1, 1, 1])

# %%
import numpy as np

np.isin(1, [1, 3])

# %%
np.arctan2(3, 1)
# %%
np.arctan2(1, 1)
# %%
np.arctan2(-1, 1)
# %%
np.arctan2(1, -1)
# %%
np.arctan2(-1, -1)
# %%
# foot0_range = np.array([4.18879, 0.5236])
# foot1_range = np.array([5.75959, 2.0944])

# foot0_range = np.array([-2.0944, 0.5236])
# foot1_range = np.array([-0.5236, 2.0944])

# foot2_range = np.array([1.0472, 3.6652])
# foot3_range = np.array([2.6180, 5.23599])

# %%
point = np.array([0.02, 0.02])

reach_range = [0.08, 0.35]
feet_ranges = [
    np.array([-2.0944, 0.5236]),
    np.array([-0.5236, 2.0944]),
    np.array([1.0472, 3.6652]),
    np.array([2.6180, 5.23599]),
]

id:int = 1
angle = np.arctan2(point[1], point[0])
print(angle)
orig_angle = angle

if (id == 2) or (id == 3):
    if angle < 0:
        angle = angle + 2 * np.pi
    
print(angle)

test = ((angle > feet_ranges[id][0]) & (angle < feet_ranges[id][1]))
print(test)

if angle < feet_ranges[id][0]:
    angle = feet_ranges[id][0]
elif angle > feet_ranges[id][1]:
    angle = feet_ranges[id][1]
print(angle)

dist = np.linalg.norm(point, 2)
print(dist)

if dist > reach_range[1]:
    dist = reach_range[1]
elif dist < reach_range[0]:
    dist = reach_range[0]
print(dist)

# - use angle and dist to make new point
x = dist * np.cos(angle)
y = dist * np.sin(angle)
point = np.array([x, y])
print(point)

print(np.arctan2(y, x))
print(np.linalg.norm(point))

# %%
import numpy as np

test = np.array([1, 2, 3, 4])
deleted = np.delete(test, 0)
print(deleted)

# %%
action_idx = 1
legs = np.delete(np.array([0, 1, 2, 3]), action_idx)
walk_order = np.random.permutation(legs)
print(walk_order)

# %%
import numpy as np

class gaussian (object):
    def __init__(self, origin:np.array, sigma:float) -> None:
        self.x0 = origin[0]
        self.y0 = origin[1]
        self.sigma = sigma
    
    def eval (self, location:np.array) -> float:
        return (1 / (2 * np.pi * self.sigma**2)) * np.exp(-1 * ((location[0] - self.x0)**2 + (location[1] - self.y0)**2) / (2 * self.sigma**2))

# %%
from stable_baselines3 import PPO
from stable_baselines3.common.envs import SimpleMultiObsEnv


# Stable Baselines provides SimpleMultiObsEnv as an example environment with Dict observations
env = SimpleMultiObsEnv(random_start=False)

# %%
model = PPO("MultiInputPolicy", env, verbose=1)

# %%
model.learn(total_timesteps=100_000)

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('labeled_objects.png')

# %%
robot = np.array([120, 100])
goal = np.array([200, 250])
im_robot = cv2.circle(im, robot, radius=30, color=(0, 0, 255), thickness=-1)
im_goal = cv2.circle(im_robot, goal, radius=30, color=(0, 255, 0), thickness=-1)
plt.imshow(im_goal)

# %%
# TODO try to crop out the subset of the image closest to the robot
res = 100
crop_img = im_goal[robot[1] - res:robot[1] + res, robot[0] - res:robot[0] + res]

# %%
plt.imshow(crop_img)

# %%
