import numpy as np
import matplotlib.pyplot as plt
import math

t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * 5 * t)
xf = x * np.exp(-1j * 2 * np.pi * t)
xf3 = x * np.exp(-1j * 2 * np.pi * 3 * t)
xf5 = x * np.exp(-1j * 2 * np.pi * 5 * t)
xf7 = x * np.exp(-1j * 2 * np.pi * 7 * t)
xf11 = x * np.exp(-1j * 2 * np.pi * 11 * t)

signals = [x, xf, xf3, xf5, xf7, xf11]
titles = ['base', '1', '3', '5', '7', '11']

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

for i, sig in enumerate(signals):
    if np.iscomplexobj(sig):
        axes[i].plot(np.real(sig), np.imag(sig))
    else:
        axes[i].set_ylim(-1, 1)
        axes[i].plot(sig)
    axes[i].set_title(titles[i])
    axes[i].grid(True)
    axes[i].axis('auto')

plt.tight_layout()
plt.show()