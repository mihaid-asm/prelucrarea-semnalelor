import numpy as np
import matplotlib.pyplot as plt
import math
import time

N = [2 ** i for i in range(7, 14)]


def fourier(n):
    four = np.zeros((n, n), dtype=complex)
    for i1 in range(n):
        for i2 in range(n):
            four[i1, i2] += np.round(math.e ** (-2 * math.pi * i1 * i2 * 1j / n), 3)
    return four

def fast_four(v):
    l = v.shape[0]
    if l <= 1:
        return v

    even = fast_four(v[0::2])
    odd = fast_four(v[1::2])

    res = np.zeros(l)
    for k in range(l // 2):
        res[k] = even[k] + np.exp(-2j * np.pi * k / l) * odd[k]
        res[k + l // 2] = even[k] - np.exp(-2j * np.pi * k / l) * odd[k]

    return res


time_dft = []
time_fft = []
time_npfft = []

for size in N:
    rand_arr = np.random.randn(size)
    four_matr = fourier(size)
    start_d = time.perf_counter()
    four_trans = np.matmul(four_matr, rand_arr)
    end_d = time.perf_counter()
    time_dft.append(end_d - start_d)
    print(f"{size} DFT done")
    start_f = time.perf_counter()
    four_fft = fast_four(rand_arr)
    end_f = time.perf_counter()
    time_fft.append(end_f - start_f)
    print(f"{size} FFT done")
    start_n = time.perf_counter()
    four_npfft = np.fft.fft(rand_arr)
    end_n = time.perf_counter()
    time_npfft.append(end_n - start_n)
    print(f"{size} NP.FFT done")

plt.plot(time_dft, label='DFT')
plt.plot(time_fft, label='FFT')
plt.plot(time_npfft, label='NP.FFT')

plt.yscale('log')
plt.legend()
plt.show()