import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

steps = 200

def fourier_weighting(series, steps):
    preds = np.zeros(steps)
    four = np.abs(np.fft.fft(series))[1:steps+1]
    denom = np.sum(four)
    for i in range(steps):
        past = np.concatenate((np.flip(preds[:i]), series[-1:-(steps-i)-1:-1]))
        numer = four @ past
        preds[i] = numer / denom
    return preds

consumption = pd.read_csv("../household_hourly_dataset.csv")
power = np.array(consumption["Global_active_power"])
split = int(np.round(len(power) * 0.7))
train_p = power[:split]
ground_truth = power[split:split+steps]
preds = fourier_weighting(train_p, steps)

plt.figure(figsize=(12, 6))
plt.title("Transformer Results")
plt.plot(ground_truth, label="Ground Truth")
plt.plot(preds, label="Transformer Prediction", color='red', linestyle=':')
plt.legend()
plt.savefig("Fourier_weighting_pred.pdf", format="pdf")

