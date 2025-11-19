import numpy as np
import matplotlib.pyplot as plt

B = 3
freqs = [1, 1.5, 2, 4]
t_cont = np.linspace(-3, 3, 6000)

def sampling(f_s):
    a = np.arange(0, -3-1e-9, -1/f_s)[::-1]
    b = np.arange(0, 3+1e-9, 1/f_s)[1:]
    return np.concatenate([a, b])

def sinc_reconstruct(t_cont, t_samp, x_samp, Ts):
    x_rec = np.zeros_like(t_cont)
    for n, x_n in enumerate(x_samp):
        x_rec += x_n * np.sinc((t_cont - t_samp[n]) / Ts)
    return x_rec

fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=False)

for i in range(4):
    row = i // 2
    col = i % 2
    t = sampling(freqs[i])
    sinc = np.sinc(B*t_cont)
    sincstem = np.sinc(B*t)
    x_hat = sinc_reconstruct(t_cont, t, sincstem, 1/freqs[i])
    axs[row][col].plot(t_cont, sinc)
    axs[row][col].stem(t, sincstem)
    axs[row][col].plot(t_cont, x_hat, linestyle="dashed")

plt.tight_layout()
plt.show()