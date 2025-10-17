import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice as sd

fs = 44100

t = np.linspace(0, 5, int(fs*5))
a = np.cos(800*np.pi*t + np.pi/3)
b = np.sin(300*np.pi*t + np.pi/2)
c = a + b
c_int16 = np.int16(c / np.max(np.abs(c)) * 32767)
scipy.io.wavfile.write('combo.wav', fs, c_int16)
ratee, x = scipy.io.wavfile.read('combo.wav')
sd.play(x, ratee)
sd.wait()