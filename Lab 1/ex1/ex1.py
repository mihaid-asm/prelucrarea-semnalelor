import numpy as np
import matplotlib.pyplot as plt

t0 = np.linspace(0,0.03, int(0.03 / 0.00005))
t = np.linspace(0,0.03, int(200 * 0.03))
x0 = np.cos(520*np.pi*t0 + np.pi/3)
y0 = np.cos(280*np.pi*t0 - np.pi/3)
z0 = np.cos(120*np.pi*t0 + np.pi/3)

plt.plot(t0, x0, label="x(t)")
plt.show()
plt.plot(t0, y0, label="x(t)")
plt.show()
plt.plot(t0, z0, label="x(t)")
plt.show()

x = np.cos(520*np.pi*t + np.pi/3)
y = np.cos(280*np.pi*t - np.pi/3)
z = np.cos(120*np.pi*t + np.pi/3)

plt.stem(t, x, label="x(t)")
plt.show()
plt.stem(t, y, label="x(t)")
plt.show()
plt.stem(t, z, label="x(t)")
plt.show()
