import numpy as np

print(np.load("scores-baseline.npy").mean())
print(np.load("scores-both.npy").mean())
print(np.load("scores-unigram.npy").mean())
