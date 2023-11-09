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

