import matplotlib.pyplot as plt
import cv2.cv2 as cv2

from hw4_functions import *


def main():
    print("----------------------------------------------------\n")
    print_IDs()

    print("-----------------------image 1----------------------\n")
    im1 = cv2.imread(r'Images\baby.tif')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1_clean = clean_im1(im1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im1_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("The images are noised by S&P noise and should be transform to the original shape.\n"
          "We first cleaned the S&P noise using a median filter.\n"
          "Then we created an projective transform and mapped each picture, then we average them.\n")

    print("-----------------------image 2----------------------\n")
    im2 = cv2.imread(r'Images\windmill.tif')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2_clean = clean_im2(im2)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im2_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("The image has a peak (high value) in one of its frequencies.\n"
          "We calculated the image's fft, found the peaks and zeroed them.\n")

    print("-----------------------image 3----------------------\n")
    im3 = cv2.imread(r'Images\watermelon.tif')
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    im3_clean = clean_im3(im3)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im3_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("The image is blurred.\n"
          "We calculated its highpass filter and added it to the image "
          "which emphasized the edges and sharpened the image.\n")

    print("-----------------------image 4----------------------\n")
    im4 = cv2.imread(r'Images\umbrella.tif')
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
    im4_clean = clean_im4(im4)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im4, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im4_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("The picture is shifting (echo image).\n"
          "We moved to fft space and then used the formula from tutorial 7 "
          "(a, b are known by over the mouse on the picture).\n")

    print("-----------------------image 5----------------------\n")
    im5 = cv2.imread(r'Images\USAflag.tif')
    im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
    im5_clean = clean_im5(im5)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im5, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im5_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print(f"The image is noised by writing which disturb to the horizontal stripes.\n"
          f"The cleaning process divided to 3 sub-images: 1. stars, 2. right to stars, 3.down to stars.\n"
          f"1. stars didn't change, the others - we saved the columns from image's fft (0, u) and zeroed the others.\n")

    print("-----------------------image 6----------------------\n")
    im6 = cv2.imread(r'Images\cups.tif')
    im6 = cv2.cvtColor(im6, cv2.COLOR_BGR2GRAY)
    im6_clean = clean_im6(im6)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im6, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im6_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("The picture is dark and has ringing.\n"
          "When we moved to fft space we noticed a dark square in the center (low frequencies) "
          "which didn't seem fit to the area.\n"
          "We made it brighter without change the DC.\n")

    print("-----------------------image 7----------------------\n")
    im7 = cv2.imread(r'Images\house.tif')
    im7 = cv2.cvtColor(im7, cv2.COLOR_BGR2GRAY)
    im7_clean = clean_im7(im7)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im7, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im7_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("The picture is shifting (echo image) 10 times.\n"
          "we moved to fft space and then used the formula from tutorial 7"
          " (a, b are known by over the mouse on the picture).\n"
          "The black box in the top left corner helped us to get a, b.\n"
          "Then we mapped its colors to [0, 255] which enhanced the contrast.\n")

    print("-----------------------image 8----------------------\n")
    im8 = cv2.imread(r'Images\bears.tif')
    im8 = cv2.cvtColor(im8, cv2.COLOR_BGR2GRAY)
    im8_clean = clean_im8(im8)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im8, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im8_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("The image is very dark, we mapped its colors to [0, 255] which enhanced the contrast.\n")

    plt.show()


if __name__ == "__main__":
    main()
