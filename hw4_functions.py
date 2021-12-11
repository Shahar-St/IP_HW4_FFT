import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2


def print_IDs():
    print("305237257+312162027\n")


def clean_im1(im):
    clean_im = 0
    return clean_im


def clean_im2(im):
    clean_im = 0
    return clean_im


def clean_im3(im):
    clean_im = 0
    return clean_im


def clean_im4(im):
    clean_im = 0
    return clean_im


def clean_im5(im):
    clean_im = 0
    return clean_im


def clean_im6(im):
    clean_im = 0
    return clean_im


def clean_im7(im):
    clean_im = 0
    return clean_im


def clean_im8(im):
    clean_im = 0
    return clean_im


'''
    # an example of how to use fourier transform:
    img = cv2.imread(r'Images\windmill.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fourier = np.fft.fftshift(np.fft.fft2(img))
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')
    
    plt.subplot(1,3,2)
    plt.imshow(np.log(abs(img_fourier)), cmap='gray')
    plt.title('fourier transform of image')

    img_inv = np.fft.ifft2(img_fourier)
    plt.subplot(1,3,3)
    plt.imshow(abs(img_inv), cmap='gray')
    plt.title('inverse fourier of the fourier transform')

'''
