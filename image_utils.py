import numpy as np
import rawpy
import imageio

from contants import CONVERSION_MATRIX, EPSILON

DEBUG = True


def stretch(im: np.ndarray):
    return (im - np.min(im)) / np.ptp(im)


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
