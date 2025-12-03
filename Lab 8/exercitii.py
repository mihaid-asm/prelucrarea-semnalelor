import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.arange(1, N + 1, 1)

trend = t ** 2 - 3 * t + 3
season = 100000 * (np.sin(3*t) + np.sin(5*t))
noise = np.random.normal(0, 50000, N)
series = trend + season + noise

fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

axs[0].plot(t, series)
axs[0].set_title("Series")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t, trend)
axs[1].set_title("Trend")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(t, season)
axs[2].set_title("Season")
axs[2].legend()
axs[2].grid(True)

axs[3].plot(t, noise)
axs[3].set_title("Noise")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()

series_flip = np.flip(series)
series_fft = np.fft.fft(series)
series_flip_fft = np.fft.fft(series_flip)
gamma_fourier = series_fft * series_flip_fft
gamma = np.fft.ifft(gamma_fourier)
normal_gamma = np.real(gamma / gamma[0])

plt.plot(normal_gamma[:500]) # nu exista functie pentru asa ceva in numpy
plt.show()

predictions = np.zeros(N)
predictions[:100] = series[:100]
p = 10
for i in range(10, N):
    y = np.array([series[i-k] for k in range(p)]).T
    Y = np.array([[series[i-k1-k2-1] for k2 in range(p)] for k1 in range(p)])
    x = np.linalg.inv(Y.T @ Y) @ Y.T @ y
    predictions[i] = x.T @ Y[0]

plt.plot(series[:50])
plt.plot(predictions[:50], color="red", linestyle=":")
plt.show()

loss = np.zeros(51)
loss[0] = 1e6
loss[1] = 1e6


for p_ in range(2, 51):
    predictions_ = np.zeros(N)
    predictions_[:p_] = series[:p_]
    for i in range(p_, N):
        print(f"[{p_} {i}]")
        y_ = np.array([series[i - k] for k in range(p_)]).T
        Y_ = np.array([[series[i - k1 - k2 - 1] for k2 in range(p_)] for k1 in range(p_)])
        x_ = np.linalg.inv(Y_.T @ Y_) @ Y_.T @ y_
        predictions_[i] = x_.T @ Y_[0]
    loss[p_] = np.mean((predictions_ - series) ** 2)

m = np.argmin(loss)

predictionsm = np.zeros(N)
predictionsm[:m] = series[:m]
for i in range(m, N):
    ym = np.array([series[i - k] for k in range(m)]).T
    Ym = np.array([[series[i - k1 - k2 - 1] for k2 in range(m)] for k1 in range(m)])
    xm = np.linalg.inv(Ym.T @ Ym) @ Ym.T @ ym
    predictionsm[i] = xm.T @ Ym[0]
    print(f"{i}. {xm}")

plt.plot(loss)
plt.axvline(x=m, color='red', linestyle=':')
plt.yscale("log")
plt.show()
