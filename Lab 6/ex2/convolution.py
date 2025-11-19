import numpy as np
import matplotlib.pyplot as plt

times = 6
x = np.random.randn(100)
t = np.arange(-3, 3, 0.01)
rec = ((t >= -0.3) & (t <= 0.3)).astype(float)

fig, axs = plt.subplots(times, 1, figsize=(8, 6), sharex=False)

axs[0].plot(rec)

for t in range(times-1):
    x1 = rec.copy()
    x2 = rec.copy()
    s = rec.shape[0]
    rec = np.zeros(2*s)
    for i in range(s):
        for j in range(s):
            rec[i+j] += x1[i] * x2[j]
    # rec /= np.linalg.norm(rec)
    axs[t+1].plot(np.abs(rec))

plt.tight_layout()
plt.show()