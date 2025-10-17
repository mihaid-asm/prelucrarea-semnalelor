import numpy as np
import matplotlib.pyplot as plt
import math
from functools import reduce

t = np.linspace(-np.pi/2, np.pi/2, 628)

alfa = t
real_sin = np.sin(t)
taylor = np.zeros_like(t)
for n in range(50):
    power = 2*n + 1
    taylor += ((-1)**n) * t**power / math.factorial(power)
pade = (t - (7 * t ** 3) / 60) / (1 + t ** 2 / 20)

err_real = real_sin - alfa
err_tay = taylor - alfa
err_pade = pade - alfa

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=False)
plt.yscale('log')

axs[0].plot(t, alfa, label="alfa")
axs[0].plot(t, real_sin, label="real")
axs[0].plot(t, err_real, label="err")
axs[0].set_title("Real")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t, alfa, label="alfa")
axs[1].plot(t, taylor, label="taylor")
axs[1].plot(t, err_tay, label="err")
axs[1].set_title("Taylor")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(t, alfa, label="alfa")
axs[2].plot(t, pade, label="pade")
axs[2].plot(t, err_pade, label="err")
axs[2].set_title("Pade")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()