import numpy as np
import matplotlib.pyplot as plt
import scipy

t = np.linspace(0,5, 300)

s0 = np.sin(2*np.pi*t)
s1 = np.sin(2*np.pi*t + np.pi / 2)
s2 = np.sin(2*np.pi*t + np.pi)
s3 = np.sin(2*np.pi*t + 3 * np.pi / 2)

plt.plot(t, s0, label="s0")
plt.plot(t, s1, label="s1")
plt.plot(t, s2, label="s2")
plt.plot(t, s3, label="s3")

plt.show()

noise = np.random.normal(0, 1, size=300)

s0norm = np.linalg.norm(s0)
znorm = np.linalg.norm(noise)
ratios = [0.1, 1, 10, 100]
gammas = []
for r in ratios:
    gammas.append(s0norm / (znorm * r))

fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=False)

for i in range(4):
    axs[i].plot(t, s0 + gammas[i] * noise, label="S" + str(i))
    axs[i].set_title("Signal" + str(i))
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()