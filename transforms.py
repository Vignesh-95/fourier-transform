import numpy as np
import cv2 as cv


# def DFT(image):
#     new_image = np.zeros(image.shape, image.dtype)
#     for u in range(image.shape[0]):
#         for v in range(image.shape[1]):
#             sum = 0
#             for x in range(image.shape[0]):
#                 for y in range(image.shape[1]):
#                     sum += image[x, y] * np.exp(-2j * (np.pi/image.shape[0]) * (u*x+v*y))
#             sum /= image.shape[0]
#             new_image[u, v] = sum
#     return new_image

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
                               X_even + factor[int(N / 2):] * X_odd])


if __name__ == "__main__":
    img = cv.imread("/home/vignesh/PycharmProjects/Transforms_COS791_Assignment2/images/lena.png", cv.IMREAD_GRAYSCALE)
    # new_image = DFT(img)
    # new_image = np.fft.fftshift(new_image)
    # cv.imshow("", new_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    img_list = img.tolist()
    arr = []

    j = 0
    while j < img.shape[0]:
        i = 0
        arr = []
        while i < img.shape[1]:
            arr.append(img_list[j][i])
            i = i + 1
        arr = FFT(np.array(arr))
        i = 0
        while i < img.shape[1]:
            img_list[j][i] = arr[i]
            i = i + 1
        j = j + 1

    img = np.array(img_list)
    img = img.real
    cv.imshow("", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    i = 0
    while i < img.shape[1]:
        j = 0
        arr = []
        while j < img.shape[0]:
            arr.append(img_list[j][i])
            j = j + 1
        arr = FFT(np.array(arr))
        j = 0
        while j < img.shape[0]:
            img_list[j][i] = arr[i]
            j = j + 1
        i = i + 1

    img = np.array(img_list)
    img = img.real
    cv.imshow("", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
