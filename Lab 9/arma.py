import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

N = 1000
t = np.arange(1, N + 1, 1)

trend = t ** 2 - 3 * t + 3
season = 100000 * (np.sin(3*t) + np.sin(5*t))
noise = np.random.normal(0, 50000, N)
series = trend + season + noise
series /= 10000

predictions = np.zeros(N)
predictions[:100] = series[:100]
p = 10
for i in range(100, N):
    mu = np.mean(series[i-100:i])
    y = np.array([series[i-k] for k in range(p+1)]).T
    Y = np.array([[np.abs(series[i-k1-k2-1] - mu) for k2 in range(p)] for k1 in range(p+1)])
    Y[-1, :] = mu * np.ones(p)
    x = np.linalg.inv(Y.T @ Y) @ Y.T @ y
    predictions[i] = x.T @ Y[0]

plt.plot(series[:200])
plt.plot(predictions[:200], color="red", linestyle=":")
plt.savefig("ma.pdf", format="pdf")

train = series[:50]
test = series[50:60]
all = series[:60]
model = ARIMA(train, order=(4, 2, 5))
model_fit = model.fit()

result = model_fit.get_forecast(steps=10)
mean = result.predicted_mean
conf_int = result.conf_int()

plt.figure(figsize=(10,5))
plt.plot(all, label='Training Data')
plt.plot(np.arange(50, 60, 1), mean, label='Forecast', color='red', linestyle=':')
plt.legend()
plt.savefig("arima.pdf", format="pdf")