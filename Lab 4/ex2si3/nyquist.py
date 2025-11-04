import numpy as np
import matplotlib.pyplot as plt

f = 58
f_s = 95
stop = 0.1
ta = np.arange(0, stop, 1/(2 * (f + 2 * f_s) + 10))  # am folosit doar fs pentru aliere(ex2)
tplot = np.linspace(0, stop, 3000)
a = np.sin(2*f*np.pi*ta)
aplot = np.sin(2*f*np.pi*tplot)
b = np.sin(2*(f+f_s)*np.pi*ta)
bplot = np.sin(2*(f+f_s)*np.pi*tplot)
c = np.sin(2*(f+2*f_s)*np.pi*ta)
cplot = np.sin(2*(f+2*f_s)*np.pi*tplot)

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

axs[0].plot(tplot, aplot, label="A")
axs[0].stem(ta, a)
axs[0].set_title("Signal A")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(tplot, bplot, label="B")
axs[1].stem(ta, b)
axs[1].set_title("Signal B")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(tplot, cplot, label="C")
axs[2].stem(ta, c, label="C")
axs[2].set_title("Signal C")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()