import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import scipy as sp

ws = [5, 9, 13, 17]
i = 1

x = pd.read_csv("train.csv")["Count"][:72]
samples = x.shape[0]

fig, axs = plt.subplots(len(ws) + 1, 1, figsize=(8, 6), sharex=False)

axs[0].plot(x)

b, a = sp.signal.cheby1(7, 10, 0.75, btype='low', analog=False)
# as alege cheby pentru ca la inceput pare sa fie mai netezit decat butter

for w in ws:
    xc = np.convolve(x, np.ones(w), 'valid') / w
    xcfft = np.fft.fft(xc)
    sh = xcfft.shape[0]
    # for rem in range(sh // 8, sh - sh // 8 + 1): # indicele de taiere este 27
    #     xcfft[rem] = 0 # frecventa normalizata e 27/36 = 0.75
    # xc = np.fft.ifft(xcfft)
    xc = sp.signal.lfilter(b, a, xc)
    axs[i].plot(xc)
    i += 1

plt.tight_layout()
plt.show()
