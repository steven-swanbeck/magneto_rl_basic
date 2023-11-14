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
