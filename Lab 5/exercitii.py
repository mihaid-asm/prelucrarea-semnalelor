import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

x = pd.read_csv("Cars/train.csv")
samples = x.shape[0]

f_s = 1/3600
print(f"a) f = {f_s}")

start_time = x["Datetime"][0]
end_time = x["Datetime"][samples-1]
print(f"b) Between {start_time} and {end_time}")

print(f"c) Ca sa nu se faca aliere, f_max = ({round(f_s, 6)}/2)Hz = {round(f_s/2, 6)}Hz")

xc = np.array(x["Count"])
xc = xc - np.mean(xc)
X = np.abs(np.fft.fft(xc))[:samples // 2]

top4 = np.argsort(X)[-1:-5:-1]
print(f"f) Top 4: {top4}") # o ora, 2 ore, aproximativ o luna, 3 ore

X_top4 = np.zeros(samples)
for i in top4:
    X_top4[i] = X[i]
    X_top4[-i] = X[-i]

xsleek = np.abs(np.fft.ifft(X_top4))

month = 31 * 24

X_no_high = X.copy()
for i in top4:
    X_no_high[i] = 0



fft_mean = np.mean(X)
f = f_s*np.linspace(0, samples // 2, samples // 2) / samples
plt.yscale("log")
plt.plot(f, X)
plt.show()
plt.plot(X_top4)
plt.xlim((0, 5))
plt.show()
plt.plot(xsleek)
plt.show()
plt.plot(x["Count"].iloc[1056:1056+month].values)
plt.show()
plt.plot(X_no_high)
plt.show()

# PENTRU PUNCTUL H
# o idee ar fi sa ne folosim de esantioanele cu valorile cele mai mari
# si sa le asociem cu anumite perioade ale anului (eg. in apropierea sarbatorilor)
# apoi sa incercam sa dam o anumita astfel de data primului element de frecventa mare
# si sa shiftam data pana cand ajungem la primul esantion