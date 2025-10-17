import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 5, 400)
a = np.cos(800*np.pi*t + np.pi/3)
b = (480*t - np.floor(480*t))
c = a + b

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

axs[0].plot(t, a, label="A")
axs[0].set_title("Signal A")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t, b, label="B")
axs[1].set_title("Signal B")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(t, c, label="C")
axs[2].set_title("Signal C")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()