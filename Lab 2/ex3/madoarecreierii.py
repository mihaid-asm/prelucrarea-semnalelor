import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice as sd

fs = 44100

ta = np.linspace(0, 3, int(fs*3))
a = np.cos(800*np.pi*ta + np.pi/3)
a_int16 = np.int16(a / np.max(np.abs(a)) * 32767)
scipy.io.wavfile.write('subpunct_a.wav', fs, a_int16)

tb = np.linspace(0, 3, int(fs*3))
b = np.cos(1600*np.pi*tb + np.pi/3)
b_int16 = np.int16(b / np.max(np.abs(b)) * 32767)
scipy.io.wavfile.write('subpunct_b.wav', fs, b_int16)

tc = np.linspace(0, 3, int(fs*3))
c = (480*tc - np.floor(480*tc))
c_int16 = np.int16(c / np.max(np.abs(c)) * 32767)
scipy.io.wavfile.write('subpunct_c.wav', fs, c_int16)

td = np.linspace(0, 3, int(fs*3))
d = 1 - (np.floor(td * 600) % 2)
d_int16 = np.int16(d / np.max(np.abs(d)) * 32767)
scipy.io.wavfile.write('subpunct_d.wav', fs, d_int16)

ratee, x = scipy.io.wavfile.read('subpunct_d.wav')
sd.play(x, ratee)
sd.wait()