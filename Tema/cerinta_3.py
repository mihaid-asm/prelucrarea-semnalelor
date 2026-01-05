import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

Q_jpeg = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]])

def show_image(title,image,ratio=0.17):
    image = cv.resize(image,(0,0),fx=ratio,fy=ratio)
    cv.imshow(title,image[:, :, ::-1])
    cv.waitKey(0)
    cv.destroyAllWindows()

X = np.array(plt.imread("titi.jpg"))
if X.max() <= 1.0:
    X = (X * 255).astype(np.uint8)
else:
    X = X.astype(np.uint8)
X = cv.resize(X, (512, 512))[:, :, :3]
conversion = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]
])
conversion_inv = np.linalg.inv(conversion)

def RGB2YCBCR(X):
    Y = np.zeros((512, 512, 3))
    for i in range(512):
        for j in range(512):
            Y[i, j] = np.floor((conversion @ X[i, j].T).T + np.array([0, 128, 128])).astype("uint8")
    return Y

def YCBCR2RGB(X):
    Y = np.zeros((512, 512, 3))
    for i in range(512):
        for j in range(512):
            Y[i, j] = (conversion_inv @ (X[i, j].T - np.array([0, 128, 128]))).T
    return Y

def alpha_optimization(X, mse_thr):
    min_alpha = 0.1
    max_alpha = 100
    iter = 0

    conv_X = RGB2YCBCR(X)
    Y = conv_X[:, :, 0]
    Cb = conv_X[:, :, 1]
    Cr = conv_X[:, :, 2]
    Ycos = np.zeros((512, 512))
    Cbcos = np.zeros((512, 512))
    Crcos = np.zeros((512, 512))

    for i in range(0, 512, 8):
        for j in range(0, 512, 8):
            Yc = dctn(Y[i:i+8, j:j+8])
            Cbc = dctn(Cb[i:i+8, j:j+8])
            Crc = dctn(Cr[i:i+8, j:j+8])
            Ycos[i:i+8, j:j+8] = Yc
            Cbcos[i:i+8, j:j+8] = Cbc
            Crcos[i:i+8, j:j+8] = Crc

    while True:
        Yq = np.zeros((512, 512))
        Cbq = np.zeros((512, 512))
        Crq = np.zeros((512, 512))
        Yd = np.zeros((512, 512))
        Cbd = np.zeros((512, 512))
        Crd = np.zeros((512, 512))
        mid_alpha = (min_alpha + max_alpha) / 2
        Qa_jpeg = Q_jpeg * mid_alpha
        for i in range(0, 512, 8):
            for j in range(0, 512, 8):
                Yq[i:i+8, j:j+8] = np.floor(Ycos[i:i+8, j:j+8] / Qa_jpeg) * Qa_jpeg
                Cbq[i:i + 8, j:j + 8] = np.floor(Cbcos[i:i + 8, j:j + 8] / Qa_jpeg) * Qa_jpeg
                Crq[i:i + 8, j:j + 8] = np.floor(Crcos[i:i + 8, j:j + 8] / Qa_jpeg) * Qa_jpeg
                Yd[i:i+8, j:j+8] = idctn(Yq[i:i+8, j:j+8])
                Cbd[i:i + 8, j:j + 8] = idctn(Cbq[i:i + 8, j:j + 8])
                Crd[i:i + 8, j:j + 8] = idctn(Crq[i:i + 8, j:j + 8])

        decoded_X = np.dstack((Yd, Cbd, Crd))
        decoded_X = YCBCR2RGB(decoded_X)

        MSE = np.mean((X - decoded_X) ** 2)
        if MSE < mse_thr:
            min_alpha = mid_alpha
            mid_alpha = (min_alpha + max_alpha) / 2
        else:
            max_alpha = mid_alpha
            mid_alpha = (min_alpha + max_alpha) / 2

        iter += 1
        if iter >= 20 and MSE <= mse_thr:
            return mid_alpha

converted = RGB2YCBCR(X)
Y = converted[:, :, 0]
Cb = converted[:, :, 1]
Cr = converted[:, :, 2]

thr = float(input("MSE_threshold = "))

Q_down = 10

X_jpeg = X.copy()
X_jpeg = Q_down*np.round(X_jpeg/Q_down);

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
        p1 = None
        m2 = 2**31-4
        p2 = None
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

alpha = alpha_optimization(X, thr)
Qa_jpeg = Q_jpeg * alpha

for i in range(0, 512, 8):
    for j in range(0, 512, 8):
        for channel in range(3):
            x = converted[i:i+8, j:j+8, channel]
            y = dctn(x)
            y_jpeg = np.floor(y / Qa_jpeg)
            zig = zigzag(y_jpeg)
            pairs = rle(zig)
            for pair in pairs:
                t_pair = (pair[0], pair[1])
                freqs[t_pair] = freqs.get(t_pair, 0) + 1

            x_jpeg = idctn(Qa_jpeg * y_jpeg)
            y_nnz = np.count_nonzero(y)
            y_nnz_total += y_nnz
            y_jpeg_nnz = np.count_nonzero(y_jpeg)
            y_jpeg_nnz_total += y_jpeg_nnz
            quant_cos[i:i+8, j:j+8] = x_jpeg

codes = huffman(freqs)
# print(codes)

print('Componente în frecvență: ' + str(y_nnz_total) +
              '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz_total))


def snr(original, with_noise):
    original = original.astype(float)
    with_noise = with_noise.astype(float)

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
            y_jpeg = np.floor(y / Qa_jpeg)
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
            dequant = Qa_jpeg * cos
            block = idctn(dequant)
            X[8*i:8*(i+1), 8*j:8*(j+1)] = block

    return X


coded_Y = full_encoding(Y)
decoded_Y = full_decoding(coded_Y)
coded_Cb = full_encoding(Cb)
decoded_Cb = full_decoding(coded_Cb)
coded_Cr = full_encoding(Cr)
decoded_Cr = full_decoding(coded_Cr)

coded_X = (coded_Y, coded_Cb, coded_Cr)
decoded_X = YCBCR2RGB(np.dstack((decoded_Y, decoded_Cb, decoded_Cr)))

bytes = 0
for component in coded_X:
    for row in component:
        for cod in row:
            bytes += len(cod)
print(f"Total Bits: {bytes}")

print(f"SNR: {snr(X, decoded_X)}")

print(f"Alpha: {alpha}")

print(f"MSE: {np.mean((X-decoded_X) ** 2)}")

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')

plt.subplot(122)
plt.imshow(decoded_X / 255, cmap=plt.cm.gray)
plt.title('Encoded & Decoded')

plt.show()
