import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pywt

def show_image(title,image,ratio=0.17):
    image = cv.resize(image,(0,0),fx=ratio,fy=ratio)
    cv.imshow(title,image[:, :, ::-1])
    cv.waitKey(0)
    cv.destroyAllWindows()

X = np.array(plt.imread("poza.jpg"))
X = cv.resize(X, (512, 512))
X = X.astype(float)


def recursive_dwts(X, steps=3):
    if steps == 0:
        return [X]
    coeffs = pywt.dwt2(X, 'haar')
    LL, (LH, HL, HH) = coeffs
    return recursive_dwts(LL, steps-1) + [LH, HL, HH]


def quantization(subbands, Q=10):
    subbands[0] = subbands[0] / Q
    l = len(subbands) // 3
    for i in range(l):
        for j in range(3):
            subbands[3*i+j+1] = np.sign(subbands[3*i+j+1]) * np.floor(np.abs(subbands[3*i+j+1]) / (Q * 2 ** i))
    return subbands


def dequantization(subbands, Q=10):
    subbands[0] = subbands[0] * Q
    l = len(subbands) // 3
    for i in range(l):
        for j in range(3):
            subbands[3*i+j+1] = subbands[3*i+j+1] * (Q * 2 ** i)
    return subbands


def inverse_recursive_dwts(subbands):
    l = (len(subbands) - 1) // 3
    for i in range(l):
        coeffs = (subbands[0], (subbands[1], subbands[2], subbands[3]))
        subbands = [pywt.idwt2(coeffs, 'haar')] + subbands[4:]
    return subbands[0]


def encoding(X):
    X = X - 128
    subbands = recursive_dwts(X)
    subbands = quantization(subbands)
    return subbands


def decoding(subbands):
    subbands = dequantization(subbands)
    X = inverse_recursive_dwts(subbands)
    X = X + 128
    return X


subbands = encoding(X)
X1 = decoding(subbands)

plt.subplot(131)
plt.imshow(X, cmap="gray")
plt.title("Original Image")

plt.subplot(132)
plt.imshow(X1, cmap="gray")
plt.title("Encoded & Decoded Image")

plt.subplot(133)
plt.imshow(np.abs(X1 - X), cmap="gray")
plt.title("Absolute Diff")
plt.show()
