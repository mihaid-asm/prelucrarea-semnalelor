import numpy as np

N = 2

# p = np.concatenate([np.random.rand(N+1), np.zeros(N)])
# q = np.concatenate([np.random.rand(N+1), np.zeros(N)])
p = np.array([1, 2, 1, 0, 0])
q = np.array([1, 2, 1, 0, 0])
r = np.zeros(2*N+1)

pfft = np.fft.fft(p)
qfft = np.fft.fft(q)
r = np.real(np.fft.ifft(pfft * qfft))

print(p)
print(q)
print(r)
