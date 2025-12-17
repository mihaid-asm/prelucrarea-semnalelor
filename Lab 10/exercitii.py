import numpy as np
import matplotlib.pyplot as plt


def roots(polinom):
    polinom = polinom.astype(float)
    sz = polinom.shape[0]
    polinom /= polinom[0]
    C = np.zeros((sz-1, sz-1))
    for i in range(sz-2):
        C[i+1, i] = 1
    C[:, -1] = -np.flip(polinom[1:])
    ev, _ = np.linalg.eig(C)
    return ev


print(roots(np.array([1, -3, 4, -6, 4])))

N = 1000
t = np.arange(1, N + 1, 1)

trend = t ** 2 - 3 * t + 3
season = 100000 * (np.sin(3*t) + np.sin(5*t))
noise = np.random.normal(0, 50000, N)
series = trend + season + noise

predictions = np.zeros(N)
predictions[:50] = series[:50]
p = 10
for i in range(50, 60):
    y = np.array([series[i-k] for k in range(p)]).T
    Y = np.array([[series[i-k1-k2-1] for k2 in range(p)] for k1 in range(p)])
    x = np.linalg.inv(Y.T @ Y) @ Y.T @ y
    pol = np.zeros(11)
    pol[:10] = np.flip(-x)
    pol[10] = 1
    print(roots(pol))
    unit = np.abs(pol)
    if np.min(unit) > 1:
        print("STATIONARY")
    else:
        print("NOT STATIONARY")

    predictions[i] = x.T @ Y[0]

plt.plot(series[:60])
plt.plot(np.arange(50, 60, 1), predictions[50:60], color="red", linestyle=":")
plt.savefig("ar.pdf", format="pdf")