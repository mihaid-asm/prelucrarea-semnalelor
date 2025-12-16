import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.arange(1, N + 1, 1)

trend = t ** 2 - 3 * t + 3
season = 100000 * (np.sin(3*t) + np.sin(5*t))
noise = np.random.normal(0, 50000, N)
series = trend + season + noise

def functie_obiectiv(series, new_series):
    sz = series.shape[0] - 1
    res = 0
    for i in range(sz-1):
        res += (new_series[i] - series[i + 1]) ** 2
    return res


def mediere_exponentiala(series, alpha):
    sz = series.shape[0]
    new_series = np.zeros(sz)
    new_series[0] = series[0]
    for t in range(0, sz-1):
        for i in range(t):
            new_series[t] += series[t-i] * alpha * (1 - alpha) ** i
        new_series[t] = (1 - alpha) ** t * series[0]
    return new_series

def mediere_dubla(series, alpha, beta):
    sz = series.shape[0]
    new_series = np.zeros(sz)
    b = np.zeros(sz)
    new_series[0] = series[0]
    b[0] = series[1] - series[0]
    for i in range(1, sz):
        new_series[i] = alpha * series[i-1] + (1-alpha) * (new_series[i-1] + b[i-1])
        b[i] = beta * (new_series[i] - new_series[i-1]) + (1-beta) * b[i-1]
    return new_series

def mediere_tripla(series, alpha, beta, gamma, L):
    sz = series.shape[0]
    new_series = np.zeros(sz)
    b = np.zeros(sz)
    c = np.zeros(sz)
    new_series[0] = series[0]
    b[0] = series[1] - series[0]
    c[0] = (series[L] - series[0]) / L
    for i in range(1, sz):
        new_series[i] = alpha * (series[i] - c[i-L]) + (1-alpha) * (new_series[i-1] + b[i-1])
        b[i] = beta * (new_series[i] - new_series[i-1]) + (1-beta) * b[i-1]
        c[i] = gamma * (series[i] - new_series[i] - b[i-1]) + (1-gamma) * c[i-L]
    return new_series

alphas = np.arange(0.05, 1, 0.05)
scores = []
minimum = 0
best_alpha = 0.05

for alpha in alphas:
    err = functie_obiectiv(series, mediere_exponentiala(series, alpha))
    scores.append(err)
    if err < minimum or alpha == 0.05:
        minimum = err
        best_alpha = alpha

plt.plot(alphas, scores)
plt.axvline(x=best_alpha, color="red", linestyle=":")
plt.yscale("log")
plt.savefig("mediere-exponentiala.pdf", format="pdf")

betas = np.arange(0.05, 1, 0.05)
scores2d = np.zeros((19, 19))
minimum = 0
best_param = [0.05, 0.05]

for alpha in alphas:
    for beta in betas:
        err = functie_obiectiv(series, mediere_dubla(series, alpha, beta))
        scores2d[int(alpha * 20 - 1), int(beta * 20 - 1)] = err
        if err < minimum or (alpha == 0.05 and beta == 0.05):
            minimum = err
            best_param = [alpha, beta]

print(f"Minimum double [alpha, beta] = {best_param} with loss = {minimum}")

fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131)
im1 = ax1.imshow(scores2d, origin='lower', extent=[0.025, 0.975, 0.025, 0.975],
                 aspect='auto', cmap='viridis')
ax1.set_xlabel('Beta')
ax1.set_ylabel('Alpha')
ax1.set_title('Heatmap (Eroare)')
fig.colorbar(im1, ax=ax1)
plt.savefig("mediere-dubla.pdf", format="pdf")

gammas = np.arange(0.05, 1, 0.05)
scores3d = np.zeros((19, 19, 19))
minimum = 0
best_param3 = [0.05, 0.05, 0.05]
for alpha in alphas:
    for beta in betas:
        for gamma in gammas:
            err = functie_obiectiv(series, mediere_tripla(series, alpha, beta, gamma, 15))
            scores3d[int(alpha * 20 - 1), int(beta * 20 - 1), int(gamma * 20 - 1)] = err
            if err < minimum or (alpha == 0.05 and beta == 0.05 and gamma == 0.05):
                minimum = err
                best_param = [alpha, beta, gamma]

print(f"Minimum triple [alpha, beta, gamma] = {best_param3} with loss = {minimum}")