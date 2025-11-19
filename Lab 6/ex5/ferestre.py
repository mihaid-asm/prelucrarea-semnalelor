import numpy as np
import matplotlib.pyplot as plt

sz = 10
step = 0.01
l = sz / step

t = np.arange(0, sz, step)
r = ((t >= 3) & (t <= 7)).astype(float)
h = 0.5 * (1 - np.cos(2 * np.pi * t / sz))
s = np.sin(np.pi * 3 * t)

sxr = s * r
sxh = s * h

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

axs[0].plot(t, s)
axs[0].set_title("Sin")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t, sxr)
axs[1].plot(t, r, color="red")
axs[1].set_title("Sin with rectangular")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(t, sxh)
axs[2].plot(t, h)
axs[2].set_title("Sin with hanning")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
