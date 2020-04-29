import numpy as np
import rawpy
import imageio
from medpy.filter import anisotropic_diffusion
from scipy.ndimage import label, binary_dilation, binary_closing
import matplotlib.pyplot as plt

from contants import CONVERSION_MATRIX, EPSILON, INTENSITY_METHOD, DELTA_THRESHOLD
from scipy.ndimage import gaussian_filter

DEBUG = True


def normalize(im: np.ndarray):
    return (im - np.min(im)) / np.ptp(im)


def linear(value, minimum, maximum):
    return (maximum - minimum) * value + minimum


def image_intensities(im: np.ndarray, method: INTENSITY_METHOD = INTENSITY_METHOD.NORM.L2):
    """

    :param im:
    :param method: Methods relying on norms. Options are [0, 1, 2, np.inf]
    :return:
    """
    # Reduce dimension to attempt to find intensity only
    norm_method = int(method.value) if method.value != np.inf else np.inf
    intensities_flash = np.linalg.norm(im, ord=norm_method, axis=2, keepdims=True)
    intensities_flash = normalize(intensities_flash)

    # Assume light intensity is continuous, so smooth but retain features.
    # Done using a anisotropic diffusion - a technique aiming at reducing image noise without removing significant
    # parts of the image content. https://en.wikipedia.org/wiki/Anisotropic_diffusion
    intensities_smoothed = anisotropic_diffusion(intensities_flash)

    return intensities_smoothed


def fill_holes(arr: np.ndarray, mask: np.ndarray, spatial_color_map: np.ndarray, hole_size_threshold=0.001):
    """

    :param arr:
    :param mask:
    :param spatial_color_map:
    :param hole_size_threshold:
    :return:
    """
    output = np.copy(arr)
    mask = np.squeeze(mask)
    labels, count = label(mask)
    sample_size = 0.01 * mask.size  # Roughly 1% of pixels are sampled
    sample_mask = normalize(np.random.uniform(size=spatial_color_map.shape[:2]) + mask.astype(np.float)) < \
                  (sample_size / (spatial_color_map.size - np.sum(mask)))
    sample_colors = spatial_color_map[sample_mask]
    hole_sizes = np.bincount(labels.ravel())[1:]  # 0's are ignored
    relevant_hole_indexes = np.where((hole_sizes / mask.size) > hole_size_threshold)[0] + 1
    all_holes = np.zeros_like(arr)
    for hole_label in relevant_hole_indexes:
        hole = labels == hole_label
        hole = np.squeeze(hole)
        average_color = np.mean(spatial_color_map[hole], axis=0)
        color_differences = np.linalg.norm(sample_colors - average_color, axis=-1)
        smallest_delta = np.min(color_differences)
        if smallest_delta > DELTA_THRESHOLD:
            continue

        most_similar_pixel = np.argmin(color_differences)
        sampled_color = arr[sample_mask][most_similar_pixel]
        output[hole] = sampled_color
        all_holes[hole] = sampled_color
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
    assert chromaticity.size == 2
    assert chromaticity.shape in [(2,), (2, 1)]
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
    return np.array((x, y, 1 - x - y))


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
    center_patch = im[x - patch_radius: x + patch_radius, y - patch_radius: y + patch_radius, :]
    mean = center_patch.mean(axis=(0, 1))
    unit_vec = np.array([1, 1, 1])
    projection_vec = unit_vec * np.dot(mean, unit_vec) / np.dot(unit_vec, unit_vec)
    chromaticity = mean / projection_vec
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


def generate_percentage_mask(intensities, percentage=0.01, smoothing_sigma=None, segmente=None):
    """

    :param intensities:
    :param percentage:
    :param smoothing_sigma:
    :param segmente:
    :return:
    """
    hist, bins = np.histogram(intensities, bins=255)
    normalized_cum_sum = np.cumsum(np.flip(hist)) / intensities.size
    intensity_threshold = np.flip(bins)[np.argmax(np.where(normalized_cum_sum < percentage))]
    mask = np.array(intensities >= intensity_threshold).astype(np.float64)

    should_smooth = smoothing_sigma != None
    if should_smooth:
        smooth_mask = gaussian_filter(mask, sigma=smoothing_sigma)
        mask = smooth_mask > 0.001

    if segmente:
        mask[::segmente, ...] = False
        mask[::, ::segmente, ...] = False

    return mask


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
    b = a / 2
    R = a * R + b

    a = np.diff(e[1])[0]
    b = a / 2
    G = a * G + b

    a = np.diff(e[2])[0]
    b = a / 2
    B = a * B + b

    colors = np.vstack((R.flatten(), G.flatten(), B.flatten())).T / 255
    h = h / np.sum(h)
    if fig is not None:
        f = plt.figure(fig)
    else:
        f = plt.gcf()
    ax = f.add_subplot(111, projection='3d')
    mxbins = np.array([M, N, O]).max()
    ax.scatter(R.flatten(), G.flatten(), B.flatten(), s=h.flatten() * (256 / mxbins) ** 3 / 2, c=colors)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
