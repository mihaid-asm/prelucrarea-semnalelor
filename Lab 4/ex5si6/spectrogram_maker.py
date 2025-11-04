import numpy as np
import matplotlib.pyplot as plt
import librosa as lb

s, fs = lb.load("aeiou.wav", sr=None)

np_s = np.array(s)
sz = s.shape[0]
width = sz // 100
ffts = []

for i in range(199):
    group = np_s[(sz*i)//200 : (sz*(i+2))//200]
    group_fft = np.abs(np.fft.fft(group[:width]))[:width//2]
    ffts.append(group_fft)

spectro = np.column_stack(ffts)
spectro_log = np.log(spectro + 1e-9)
plt.imshow(spectro_log, cmap="inferno", aspect=0.1)
plt.show()
