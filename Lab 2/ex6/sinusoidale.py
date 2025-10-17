import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice as sd

f = 100

ta = np.linspace(0, 5, int(f / 2))
tb = np.linspace(0, 5, int(f / 4))
tc = np.linspace(0, 5, 1)
a = np.sin(f*2*np.pi*ta)
b = np.sin(f*2*np.pi*tb)
c = np.sin(f*2*np.pi*tc)

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

axs[0].stem(ta, a, label="A")
axs[0].set_title("Signal A")
axs[0].legend()
axs[0].grid(True)

axs[1].stem(tb, b, label="B")
axs[1].set_title("Signal B")
axs[1].legend()
axs[1].grid(True)

axs[2].stem(tc, c, label="C")
axs[2].set_title("Signal C")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()