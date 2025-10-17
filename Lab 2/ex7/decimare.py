import numpy as np
import matplotlib.pyplot as plt

sec = 0.25

t = np.linspace(0, sec, int(1000*sec))
ta = t[0::4]
tb = ta[0::4]

o = np.sin(400*np.pi*t + np.pi / 3)
a = np.sin(400*np.pi*ta + np.pi / 3)
b = np.sin(400*np.pi*tb + np.pi / 3)

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

axs[0].stem(t, o, label="O")
axs[0].set_title("Original")
axs[0].legend()
axs[0].grid(True)

axs[1].stem(ta, a, label="A")
axs[1].set_title("A")
axs[1].legend()
axs[1].grid(True)

axs[2].stem(tb, b, label="B")
axs[2].set_title("B")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()