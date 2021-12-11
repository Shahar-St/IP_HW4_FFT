import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2


def print_IDs():
    print("305237257+312162027\n")


def clean_im1(im):
    im = np.array(im, dtype=float)
    clean_im = cleanImageMedian(im, (1, 1))
    old_points = np.array([[20, 6], [20, 111], [130, 6], [130, 111]])
    new_points = np.array([[0, 0], [0, 255], [255, 0], [255, 255]])
    transform = findAffineTransform(old_points, new_points)
    adjusted_im = mapImage(clean_im, transform, clean_im.shape)
    return adjusted_im


def clean_im2(im):
    im = np.array(im, dtype=float)
    img_fourier = np.fft.fftshift(np.fft.fft2(im))
    img_fourier[132, 156] = img_fourier[124, 100] = 0
    img_inv = np.abs(np.fft.ifft2(img_fourier))
    return img_inv


def clean_im3(im):
    im = np.array(im, dtype=float)
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    highpass = ndimage.convolve(im, kernel)
    clean_im = im + highpass

    return clean_im


def clean_im4(im):
    im = np.array(im, dtype=float)
    # (x0, y0) = ()

    clean_im = im
    return clean_im


def clean_im5(im):
    im = np.array(im, dtype=float)
    radius = (1, 2)
    clean_im = cleanImageMedian(im, radius)
    return clean_im


def clean_im6(im):
    im = np.array(im, dtype=float)
    clean_im = im
    return clean_im


def clean_im7(im):
    im = np.array(im, dtype=float)
    kernel = np.array([
        [-1, -1, 4, -1, -1],
        [-1, -1, 4, -1, -1],
    ])
    highpass = ndimage.convolve(im, kernel)
    clean_im = im + highpass

    return clean_im


def clean_im8(im):
    im = np.array(im, dtype=float)
    maxRangeList = [0, 255]
    clean_im = contrastEnhance(im, maxRangeList)
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


def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]

    # iterate over points to create x , x'
    x = []
    for i in range(0, N):
        x_point = pointsSet1[i][0]
        y_point = pointsSet1[i][1]
        x.append([x_point, y_point, 0, 0, 1, 0])
        x.append([0, 0, x_point, y_point, 0, 1])

    x_t = pointsSet2[:, 0:2].reshape(N * 2)
    T = np.matmul(np.linalg.pinv(x), x_t)
    T = np.array([
        [T[0], T[1], T[4]],
        [T[2], T[3], T[5]],
        [0, 0, 1]
    ])

    return T


def mapImage(im, T, sizeOutIm):
    new_im = np.zeros(sizeOutIm)

    # create meshgrid of all coordinates in new image [x,y]
    xx, yy = np.meshgrid(np.arange(sizeOutIm[0]), np.arange(sizeOutIm[1]))

    # add homogenous coord [x,y,1]
    target_coords = np.vstack((xx.reshape(-1), yy.reshape(-1), np.ones(xx.size)))

    # calculate source coordinates that correspond to [x,y,1] in new image
    source_coords = np.matmul(np.linalg.inv(T), target_coords)

    # normalize back to 2D
    source_coords = source_coords / source_coords[2]

    # find coordinates outside range and delete (in source and target)

    out_of_range_indices = np.any(
        (source_coords[0] > sizeOutIm[0] - 1) | (source_coords[1] > sizeOutIm[1] - 1) | (source_coords < 0), axis=0)
    source_coords = np.delete(source_coords, out_of_range_indices, axis=1)
    target_coords = np.delete(target_coords, out_of_range_indices, axis=1)

    # interpolate - bilinear
    ceil_points = np.ceil(source_coords).astype(np.int)
    floor_points = np.floor(source_coords).astype(np.int)
    NE, NW, SE, SW = im[ceil_points[0], ceil_points[1]], im[floor_points[0], ceil_points[1]], im[
        ceil_points[0], floor_points[1]], im[floor_points[0], floor_points[1]]
    delta_x = source_coords[0] - floor_points[0]
    delta_y = source_coords[1] - floor_points[1]
    S = (SE * delta_x) + (SW * (1 - delta_x))
    N = (NE * delta_x) + (NW * (1 - delta_x))
    V = (N * delta_y) + (S * (1 - delta_y))

    new_im[target_coords[0].astype(int), target_coords[1].astype(int)] = V
    return new_im


def cleanImageMedian(im, radius):
    median_im = im.copy()
    im = im.astype(float)

    vertical_radius, horizontal_radius = radius[0], radius[1]

    for ix in range(vertical_radius, im.shape[0] - vertical_radius):
        for iy in range(horizontal_radius, im.shape[1] - horizontal_radius):
            window = im[ix - vertical_radius: ix + vertical_radius + 1,
                     iy - horizontal_radius: iy + horizontal_radius + 1]
            median = np.median(window)
            median_im[ix, iy] = median

    median_im = median_im.astype(np.uint8)
    return median_im


def contrastEnhance(im, im_range):
    min_im_val = np.min(im)
    max_im_val = np.max(im)

    min_target_val = im_range[0]
    max_target_val = im_range[1]

    a = (max_target_val - min_target_val) / (max_im_val - min_im_val)
    b = min_target_val - (min_im_val * a)

    nim = np.copy(im)
    for (x, y), value in np.ndenumerate(im):
        nim[x][y] = a * value + b
    return nim
