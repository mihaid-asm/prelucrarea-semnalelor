import numpy as np
import matplotlib.pyplot as plt
import math

N = 64

fourier8 = np.zeros((N, N), dtype=complex)
for i1 in range(N):
    for i2 in range(N):
        fourier8[i1, i2] += math.e ** (-2 * math.pi * i1 * i2 * 1j / N)

print(fourier8)

fourier8H = np.conjugate(fourier8).T
nI = np.matmul(fourier8H, fourier8)
print(nI)
print(np.allclose(nI, N*np.eye(N)))

n = np.arange(N)
k = n.reshape((N, 1))
F = np.exp(-2j * np.pi * k * n / N)

fig, axes = plt.subplots(5, 1, figsize=(10, 12))

axes[0].plot(np.real(F[0, :]))
axes[0].plot(np.imag(F[0, :]))
axes[0].set_title(f"Linia {0}")
axes[0].grid(True)

axes[1].plot(np.real(F[1, :]))
axes[1].plot(np.imag(F[1, :]))
axes[1].set_title(f"Linia {1}")
axes[1].grid(True)

axes[2].plot(np.real(F[-1, :]))
axes[2].plot(np.imag(F[-1, :]))
axes[2].set_title(f"Linia {63}")
axes[2].grid(True)

axes[3].plot(np.real(F[2, :]))
axes[3].plot(np.imag(F[2, :]))
axes[3].set_title(f"Linia {2}")
axes[3].grid(True)

axes[4].plot(np.real(F[-2, :]))
axes[4].plot(np.imag(F[-2, :]))
axes[4].set_title(f"Linia {62}")
axes[4].grid(True)

plt.tight_layout()
plt.show()