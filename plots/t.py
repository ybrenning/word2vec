import numpy as np
import matplotlib.pyplot as plt

t = 1e-3

x = np.linspace(0, 0.1)
y = 1 - np.sqrt(t/x)

p = (np.sqrt(x/t) + 1) * (t/x)

plt.plot(x, p)
plt.show()
