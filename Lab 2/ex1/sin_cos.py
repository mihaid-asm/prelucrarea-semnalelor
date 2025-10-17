import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,5, 300)

xsin = np.sin(2*np.pi*t)
xcos = np.cos(2*np.pi*t + 3 * np.pi / 2)

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

axs[0].plot(t, xsin, label="XSIN")
axs[0].set_title("XSIN")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t, xcos, label="XCOS")
axs[1].set_title("XCOS")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()