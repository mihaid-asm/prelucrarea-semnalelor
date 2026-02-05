import numpy as np
import matplotlib.pyplot as plt

N = 999
t = np.arange(1, N + 1, 1)

trend = t ** 2 - 3 * t + 3
season = 100000 * (np.sin(3*t) + np.sin(5*t))
noise = np.random.normal(0, 50000, N)
series = trend + season + noise

L = 500
X = np.zeros((N-L+1, L))
for i in range(N-L+1):
    for j in range(L):
        X[i][j] = series[i+j]

XXT = X @ X.T
XTX = X.T @ X

eigvalxxt, eigvecxxt = np.linalg.eigh(XXT)
eigvalxtx, eigvecxtx = np.linalg.eigh(XTX)
U, S, V = np.linalg.svd(X, full_matrices=False)

idx_xtx = np.argsort(eigvalxtx)[::-1]
sorted_eigvalxtx = eigvalxtx[idx_xtx]
print(np.allclose(S**2, sorted_eigvalxtx[:len(S)]))
# S^2 = eigval(XXT) = eigval(XTX)

idx_xxt = np.argsort(eigvalxxt)[::-1]
sorted_eigvecxxt = eigvecxxt[:, idx_xxt]
print(np.allclose(np.abs(U), np.abs(sorted_eigvecxxt[:, :U.shape[1]])))
# |U| = |eigvec(XXT)|

sorted_eigvecxtx = eigvecxtx[:, idx_xtx]
print(np.allclose(np.abs(V), np.abs(sorted_eigvecxtx[:, :V.shape[0]].T)))
# |V| = |eigvec(XTX)|

def Hankelify(X):
    L, K = X.shape
    x_hankel = np.zeros(L+K-1)
    count = np.zeros(L+K-1)
    for i in range(L):
        for j in range(K):
            x_hankel[i+j] += X[i, j]
            count[i+j] += 1
    x_hankel /= count
    return x_hankel

nr_sing = len(S)
x_pred = np.zeros((nr_sing, N))

for i in range(nr_sing):
    X_i = S[i] * np.outer(U[:, i], V[i, :])
    x_pred[i, :] = Hankelify(X_i)

x_final_pred = np.sum(x_pred, axis=0)
plt.plot(series, label="serie")
plt.plot(x_final_pred, label="predictie", color="red", linestyle=":")
plt.legend()
plt.savefig(f"rezultat.pdf", format='pdf')
plt.show()
