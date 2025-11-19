import numpy as np
import matplotlib.pyplot as plt

d = 5
s = 20

ta = np.linspace(0, 1, s)
x = np.sin(np.random.randn(s))
y = np.roll(x, d)

d1 = np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(y))
d2 = np.fft.ifft(np.fft.fft(y) / np.fft.fft(x))

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

axs[0].plot(ta, d1, label="A")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(ta, d2, label="B")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
