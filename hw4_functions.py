import numpy as np
from scipy import ndimage


def print_IDs():
    print("305237257+312162027\n")


def clean_im1(im):
    im = np.array(im, dtype=float)
    clean_im = cleanImageMedian(im, (1, 1))
    old_points_1 = np.array([[20, 6], [20, 111], [130, 6], [130, 111]])
    new_points_1 = np.array([[0, 0], [0, 255], [255, 0], [255, 255]])
    transform = findProjectiveTransform(old_points_1, new_points_1)
    adjusted_im_1 = mapImage(clean_im, transform, clean_im.shape)

    old_points_2 = np.array([[4, 180], [70, 248], [50, 121], [120, 176]])
    new_points_2 = np.array([[0, 0], [0, 255], [255, 0], [255, 255]])
    transform = findProjectiveTransform(old_points_2, new_points_2)
    adjusted_im_2 = mapImage(clean_im, transform, clean_im.shape)

    old_points_3 = np.array([[162, 77], [116, 146], [242, 132], [160, 245]])
    new_points_3 = np.array([[0, 0], [0, 255], [255, 0], [255, 255]])
    transform = findProjectiveTransform(old_points_3, new_points_3)
    adjusted_im_3 = mapImage(clean_im, transform, clean_im.shape)

    adjusted_im = np.add(adjusted_im_3, np.add(adjusted_im_1, adjusted_im_2)) / 3

    return adjusted_im


def clean_im2(im):
    img_fourier = np.fft.fftshift(np.fft.fft2(im))
    img_fourier[132, 156] = img_fourier[124, 100] = 0
    img_inv = np.abs(np.fft.ifft2(img_fourier))

    return img_inv


def clean_im3(im):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    highpass = ndimage.convolve(im, kernel)
    clean_im = im + highpass

    return clean_im


def clean_im4(im):
    x0, y0 = 4, 79
    img_fourier = np.fft.fftshift(np.fft.fft2(im))

    kernel = np.zeros((img_fourier.shape[0], img_fourier.shape[1]), dtype=float)
    kernel[0, 0] = 1
    kernel[x0, y0] = 1

    img_kernel_fourier = np.fft.fftshift(np.fft.fft2(kernel))
    img_kernel_fourier = np.where(abs(img_kernel_fourier) <= 0.01, 1, img_kernel_fourier)

    img_fourier_res = (2 * img_fourier) / img_kernel_fourier
    img_inv = np.abs(np.fft.ifft2(img_fourier_res))

    return img_inv


def clean_im5(im):
    im = np.array(im, dtype=float)

    im_right_to_stars = im[0:90, 143:]
    im_down_to_stars = im[90:, :]

    img_fourier_right_to_stars = np.fft.fftshift(np.fft.fft2(im_right_to_stars))
    kernel = np.zeros((img_fourier_right_to_stars.shape[0], img_fourier_right_to_stars.shape[1]), dtype=np.float)
    kernel[:, 78:80] = 1
    img_inv_right_to_stars = np.abs(np.fft.ifft2(img_fourier_right_to_stars * kernel))

    img_fourier_down_to_stars = np.fft.fftshift(np.fft.fft2(im_down_to_stars))
    kernel = np.zeros((img_fourier_down_to_stars.shape[0], img_fourier_down_to_stars.shape[1]), dtype=np.float)
    kernel[:, 150:152] = 1
    img_inv_down_to_stars = np.abs(np.fft.ifft2(img_fourier_down_to_stars * kernel))

    im[90:, :] = img_inv_down_to_stars
    im[0:90, 143:] = img_inv_right_to_stars

    return im


def clean_im6(im):
    img_fourier = np.fft.fftshift(np.fft.fft2(im))

    kernel = np.ones((im.shape[0], im.shape[1]), dtype=np.float)
    kernel[108: 149, 108: 149] = 2
    kernel[129, 129] = 1

    img_inv = np.abs(np.fft.ifft2(img_fourier * kernel))
    img_inv = img_inv * 0.8

    return img_inv


def clean_im7(im):
    img_fourier = np.fft.fftshift(np.fft.fft2(im))

    kernel = np.zeros((img_fourier.shape[0], img_fourier.shape[1]), dtype=float)
    kernel[0, 0:10] = 1
    img_kernel_fourier = np.fft.fftshift(np.fft.fft2(kernel))
    img_kernel_fourier = np.where(abs(img_kernel_fourier) <= 0.01, 1, img_kernel_fourier)

    img_fourier_res = (2 * img_fourier) / img_kernel_fourier
    img_inv = np.abs(np.fft.ifft2(img_fourier_res))

    maxRangeList = [0, 255]
    clean_im = contrastEnhance(img_inv, maxRangeList)

    return clean_im


# todo maybe Gamma Correction?
def clean_im8(im):
    maxRangeList = [0, 255]
    clean_im = contrastEnhance(im, maxRangeList)

    return 1.35 * clean_im


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


def findProjectiveTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]

    # iterate over points to create x , x'
    x = []
    for i in range(0, N):
        x_point = pointsSet1[i][0]
        y_point = pointsSet1[i][1]
        x_t_point = pointsSet2[i][0]
        y_t_point = pointsSet2[i][1]
        x.append([x_point, y_point, 0, 0, 1, 0, -1 * x_point * x_t_point, -1 * y_point * x_t_point])
        x.append([0, 0, x_point, y_point, 0, 1, -1 * x_point * y_t_point, -1 * y_point * y_t_point])

    x_t = pointsSet2[:, 0:2].reshape(N * 2)
    T = np.matmul(np.linalg.pinv(x), x_t)
    T = np.array([
        [T[0], T[1], T[4]],
        [T[2], T[3], T[5]],
        [T[6], T[7], 1]
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
