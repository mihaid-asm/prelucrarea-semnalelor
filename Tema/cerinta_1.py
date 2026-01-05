import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

X = np.array(plt.imread("poza.jpg"))
X = cv.resize(X, (512, 512))
"""plt.imshow(X, cmap=plt.cm.gray)
plt.show()"""

Y1 = dctn(X, type=1)
Y2 = dctn(X, type=2)
Y3 = dctn(X, type=3)
Y4 = dctn(X, type=4)
freq_db_1 = 20*np.log10(abs(Y1))
freq_db_2 = 20*np.log10(abs(Y2))
freq_db_3 = 20*np.log10(abs(Y3))
freq_db_4 = 20*np.log10(abs(Y4))

"""plt.subplot(221).imshow(freq_db_1)
plt.subplot(222).imshow(freq_db_2)
plt.subplot(223).imshow(freq_db_3)
plt.subplot(224).imshow(freq_db_4)
plt.show()"""

k = 120

Y_ziped = Y2.copy()
Y_ziped[k:] = 0
X_ziped = idctn(Y_ziped)

"""plt.imshow(X_ziped, cmap=plt.cm.gray)
plt.show()"""

Q_down = 10

X_jpeg = X.copy()
X_jpeg = Q_down*np.round(X_jpeg/Q_down);

"""plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(X_jpeg, cmap=plt.cm.gray)
plt.title('Down-sampled')
plt.show()"""

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

quant_cos = np.zeros((512, 512))
y_nnz_total = 0
y_jpeg_nnz_total = 0


def zigzag(mat):
    order = [
        0,  1,  8,  16, 9,  2,  3,  10,
        17, 24, 32, 25, 18, 11, 4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6,  7,  14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    ]
    flat = mat.flatten()
    return flat[order]


def inv_zig(zig):
    mat = np.zeros((8, 8))
    order = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    for i, (r, c) in enumerate(order):
        if i < len(zig):
            mat[r, c] = zig[i]

    return mat


def rle(zig):
    result = []
    pair = [0, 0]
    for num in zig:
        if num == 0:
            pair[0] += 1
        else:
            pair[1] = int(num)
            new_pair = pair.copy()
            result += [new_pair]
            pair[0] = 0
    return result


def inv_rle(pairs):
    result = []
    for p in pairs:
        for i in range(p[0]):
            result.append(0)
        result.append(p[1])
    tail = 64 - len(result)
    for i in range(tail):
        result.append(0)
    return result


def huffman(freqs):
    codes = {}
    for f in freqs:
        codes[f] = ""
    l = len(freqs)
    while l > 1:
        m1 = 2**31-2
        p1 = (0, 0)
        m2 = 2**31-4
        p2 = (0, 0)
        for f in freqs:
            if freqs[f] <= m1:
                m2 = m1
                m1 = freqs[f]
                p2 = p1
                p1 = f
            elif freqs[f] <= m2:
                m2 = freqs[f]
                p2 = f
        del freqs[p1]
        del freqs[p2]
        freqs[p1 + p2] = m1 + m2
        for i in range(0, len(p1), 2):
            codes[(p1[i], p1[i+1])] = "0" + codes[(p1[i], p1[i+1])]
        for i in range(0, len(p2), 2):
            codes[(p2[i], p2[i+1])] = "1" + codes[(p2[i], p2[i+1])]
        #print(freqs)
        #print(codes)
        #print(l)
        l -= 1
    return codes


def encoding(pairs, codes):
    result = ""
    for p in pairs:
        t_pair = (p[0], p[1])
        result += codes[t_pair]
    return result


def decoding(bitstream, codes):
    inv_codes = {v: k for k, v in codes.items()}
    result = []
    buffer = ""

    for bit in bitstream:
        buffer += bit
        if buffer in inv_codes:
            result.append(inv_codes[buffer])
            buffer = ""

    return result


freqs = {}

for i in range(0, 512, 8):
    for j in range(0, 512, 8):
        x = X[i:i+8, j:j+8]
        y = dctn(x)
        y_jpeg = np.round(y / Q_jpeg)
        zig = zigzag(y_jpeg)
        pairs = rle(zig)
        for pair in pairs:
            t_pair = (pair[0], pair[1])
            freqs[t_pair] = freqs.get(t_pair, 0) + 1

        x_jpeg = idctn(y_jpeg)
        y_nnz = np.count_nonzero(y)
        y_nnz_total += y_nnz
        y_jpeg_nnz = np.count_nonzero(y_jpeg)
        y_jpeg_nnz_total += y_jpeg_nnz
        quant_cos[i:i+8, j:j+8] = x_jpeg
        """print('Componente în frecvență: ' + str(y_nnz) +
              '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))
        plt.subplot(121).imshow(x, cmap=plt.cm.gray)
        plt.title('Original')
        plt.subplot(122).imshow(x_jpeg, cmap=plt.cm.gray)
        plt.title('JPEG')
        plt.show()"""

codes = huffman(freqs)
# print(codes)

print('Componente în frecvență: ' + str(y_nnz_total) +
              '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz_total))

"""plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.subplot(122).imshow(quant_cos, cmap=plt.cm.gray)
plt.show()"""


def snr(original, with_noise):
    s_power = np.sum(original ** 2)
    n_power = np.sum((original - with_noise) ** 2)
    return np.real(10 * np.log10(s_power / n_power))


B = np.array([
    [-26, -3, -6,  2,  2, -1,  0,  0],
    [0, -2, -4,  1,  1,  0,  0,  0],
    [-3,  1,  5, -1, -1,  0,  0,  0],
    [-3,  1,  2, -1,  0,  0,  0,  0],
    [1,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0]
])

def full_encoding(X):
    result = []
    for i in range(0, 512, 8):
        result.append([])
        for j in range(0, 512, 8):
            x = X[i:i + 8, j:j + 8]
            y = dctn(x)
            y_jpeg = np.round(y / Q_jpeg)
            zig = zigzag(y_jpeg)
            pairs = rle(zig)
            result[i // 8].append(encoding(pairs, codes))
    return result


def full_decoding(coded):
    X = np.zeros((512, 512))
    for i in range(64):
        for j in range(64):
            pairs = decoding(coded[i][j], codes)
            zig = inv_rle(pairs)
            cos = inv_zig(zig)
            dequant = Q_jpeg * cos
            block = idctn(dequant)
            X[8*i:8*(i+1), 8*j:8*(j+1)] = block

    return X

coded_X = full_encoding(X)
decoded_X = full_decoding(coded_X)
bytes = 0
for row in coded_X:
    for cod in row:
        bytes += len(cod)
print(f"Total Bits: {bytes}")

print(f"SNR: {snr(X, decoded_X)}")

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')

plt.subplot(122)
plt.imshow(decoded_X, cmap=plt.cm.gray)
plt.title('Encoded & Decoded')

plt.show()
