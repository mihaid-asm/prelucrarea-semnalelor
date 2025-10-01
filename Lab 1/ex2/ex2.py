import numpy as np
import matplotlib.pyplot as plt

ta = np.linspace(0, 0.05, int(1600 * 0.05))
a = np.cos(800*np.pi*ta + np.pi/3)

tb = np.linspace(0, 3, 400)
b = np.cos(1600*np.pi*tb + np.pi/3)

tc = np.linspace(0, 0.05, 1600)
c = (480*tc - np.floor(480*tc))

td = np.linspace(0, 0.00005, 300)
d = 1 - (np.floor(td * 600) % 2)

fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=False)

axs[0].stem(ta, a, label="A", basefmt="k-")
axs[0].set_title("Signal A")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(tb, b, label="B")
axs[1].set_title("Signal B")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(tc, c, label="C")
axs[2].set_title("Signal C")
axs[2].legend()
axs[2].grid(True)

axs[3].plot(td, d, label="D")
axs[3].set_title("Signal D")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()

e = np.random.rand(128, 128) * 2 - 1
plt.title("Random E")
plt.imshow(e, cmap="rainbow") # rosu e spre 1, mov spre -1
plt.show()

f = np.zeros((128, 128))
for i in range(128):
    for j in range(128):
        f[i, j] = np.sin(i + j)
plt.title("Sin F")
plt.imshow(f, cmap="rainbow") # rosu e spre 1, mov spre -1
plt.show()