import numpy as np

n = 256
bias = np.random.normal(size=n)
bias /= np.linalg.norm(bias)
np.save('bias.npy', bias)
