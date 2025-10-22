import numpy as np
import matplotlib.pyplot as plt
import math

t = np.linspace(0, 1, 64)
x = np.sin(2 * np.pi * 5 * t)
y = np.cos(2 * np.pi * 8 * t + np.pi / 3)
z = np.sin(2 * np.pi * t)
xyz = x + y + z

N = 64
fourier8 = np.zeros((N, N), dtype=complex)
for i1 in range(N):
    for i2 in range(N):
        fourier8[i1, i2] += math.e ** (-2 * math.pi * i1 * i2 * 1j / N)

X = np.abs(np.matmul(fourier8, xyz))

plt.stem(X)
plt.grid(True)
plt.show()
