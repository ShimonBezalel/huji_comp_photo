import numpy as np
import rawpy
import imageio
from scipy.ndimage import label, binary_dilation, binary_closing
import matplotlib.pyplot as plt

from contants import CONVERSION_MATRIX, EPSILON
from scipy.ndimage import gaussian_filter

DEBUG = True


def stretch(im: np.ndarray):
    return (im - np.min(im)) / np.ptp(im)

def fill_holes(arr: np.ndarray, mask: np.ndarray, hole_percentage_threshold=0.001):
    """
    Taken from https://stackoverflow.com/questions/41550979/fill-holes-with-majority-of-surrounding-values-python
    :param arr:
    :return:
    """
    output = np.copy(arr)


    color_sampler = np.stack([gaussian_filter(arr[..., channel], sigma=10) for channel in range(3)], axis=2)
    if DEBUG:
        plt.imshow(color_sampler)
        plt.show()

    labels, count = label(mask)
    for idx in range(1, count + 1):
        hole = labels == idx
        is_hole_too_small = np.sum(hole)/hole.size < hole_percentage_threshold
        if is_hole_too_small:
            continue
        hole_edges = binary_dilation(hole, iterations=50) & ~binary_dilation(hole, iterations=40) & ~mask
        if DEBUG:
            plt.imshow(np.squeeze(hole_edges.astype(np.float)))
            plt.show()

        most_common_color = get_most_common_color(color_sampler, hole_edges)
        blending_mask = gaussian_filter(hole.astype(np.float), sigma=10)
        if DEBUG:
            plt.imshow(blending_mask)
            plt.show()
        color_mask = most_common_color * np.ones_like(color_sampler)
        output = color_mask * blending_mask[..., np.newaxis] + output * (1-blending_mask)[..., np.newaxis]
        # output[blending_mask > 0.00001] = blending_mask[..., np.newaxis] * most_common_color + \
        #                                   (1 - blending_mask)[blending_mask > 0.00001][..., np.newaxis] * color_sampler[blending_mask > 0.00001]
        # output[hole] = most_common_color



    # all_counts = np.zeros(shape=(n_bins,))

        #
        #     counts, colors = np.histogram(surrounding_values, bins=100)
        #     color_dists.append(colors)
        #     best_color_index.append(np.argmax(counts))
        # best_index = np.max(common_color_index)
        # output[hole[..., channel]] = most_common_color

    return output

def get_most_common_color(arr, mask=None, blurred=False):
    if not blurred:
        color_sampler = np.stack([gaussian_filter(arr[..., channel], sigma=10) for channel in range(3)], axis=2)
    else:
        color_sampler = arr
    if mask is not None:
        color_sampler = color_sampler[mask]
    n_bins_3d = 4
    hist_3d, edges = np.histogramdd(color_sampler.reshape(-1, 3),
                                    bins=[np.linspace(np.min(color_sampler[..., channel]),
                                                      np.max(color_sampler[..., channel]),
                                                      n_bins_3d) for channel in range(3)])
    if DEBUG:
        histogram3dplot(hist_3d, edges)
        plt.show()
    indexes = np.unravel_index(np.argmax(hist_3d, axis=None), hist_3d.shape)
    most_common_color = np.array([e[i] for i, e in zip(indexes, edges)]).reshape((1, 1, 3))
    return most_common_color

def chromaticity_to_vector(chromaticity: np.ndarray):
    """
    Return a vector of 3 channel color from chromaticity
    :param chromaticity: [x, y] a vector on length 2
    :return: [x, y, 1] of length 3
    """
    return np.append(chromaticity, 1)


def vector_to_chromaticity(vec: np.ndarray):
    """
    Return a chromaticy of length 2 from 3 channel color
    :param vec: [x, y, z] a vector on length 3. Z may not be 0
    :return: [x, y, 1] of length 3
    """
    assert vec.shape == (3,), "The vector must have a shape of (3,), found {}".format(vec.shape)
    v = vec.copy()
    if v[2] == 0:
        v[2] = EPSILON
    normalized: np.ndarray = (v / v[2])
    return normalized[:2]


def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))


def xyz_to_lms(im: np.ndarray):
    """
    Converts a given images from XYZ format to LMS format, using Hunt-Pointer-Estevez
    :param im: np.ndarray of shape m X n X 3 color channels
    :return:
    """
    lms = im.dot(CONVERSION_MATRIX.XYZ_TO_LMS.HPE)


def xyz_to_lms(im: np.ndarray):
    """
    Converts a given images from XYZ format to LMS format, using Hunt-Pointer-Estevez
    :param im: np.ndarray of shape m X n X 3 color channels
    :return:
    """
    lms = im.dot(CONVERSION_MATRIX.XYZ_TO_LMS.HPE)

    raise Exception("not implemented")


def calculate_chromaticity(im):
    """
    Given an image of only a gray-card in the center, calculate the chromaticity of the camera's flash + setting
    :rtype: np.ndarray
    :param im: Image should be read from tiff format, an image of just a gray-card near the center of the image.
    If more than a gray-card was in the frame of the original image, the rest is ignored.
    :return: a 3-vector of values between 0 and 1 representing the delta from perfect gray.
    """
    shape = im.shape
    x, y = shape[0] // 2, shape[1] // 2
    patch_radius = 4
    center_patch = im[x - patch_radius : x + patch_radius, y - patch_radius : y + patch_radius, :]
    mean = center_patch.mean(axis=(0, 1))
    unit_vec = np.array([1, 1, 1])
    projection_vec = unit_vec * np.dot(mean, unit_vec) / np.dot(unit_vec, unit_vec)
    chromaticity = mean / projection_vec
    if DEBUG:
        print("calculated chromaticity: {}".format(chromaticity))
    return vector_to_chromaticity(chromaticity)


def save_linear_image(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)
    imageio.imsave('samples/linear.tiff', rgb)


def open_raw(path):
    """
    Opens a raw image and returns as tiff
    """
    with rawpy.imread(path) as raw:
        # rgb = raw.postprocess()
        raw_image = raw.raw_image.copy()
        return raw_image


def histogram3dplot(h, e, fig=None):
    """
    Visualize a 3D histogram

    Parameters
    ----------

    h: histogram array of shape (M,N,O)
    e: list of bin edge arrays (for R, G and B)
    """
    M, N, O = h.shape
    idxR = np.arange(M)
    idxG = np.arange(N)
    idxB = np.arange(O)

    R, G, B = np.meshgrid(idxR, idxG, idxB)
    a = np.diff(e[0])[0]
    b = a/2
    R = a * R + b

    a = np.diff(e[1])[0]
    b = a/2
    G = a * G + b

    a = np.diff(e[2])[0]
    b = a/2
    B = a * B + b

    colors = np.vstack((R.flatten(), G.flatten(), B.flatten())).T/255
    h = h / np.sum(h)
    if fig is not None:
        f = plt.figure(fig)
    else:
        f = plt.gcf()
    ax = f.add_subplot(111, projection='3d')
    mxbins = np.array([M,N,O]).max()
    ax.scatter(R.flatten(), G.flatten(), B.flatten(), s=h.flatten()*(256/mxbins)**3/2, c=colors)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')