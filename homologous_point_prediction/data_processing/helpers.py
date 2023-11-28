import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mri_histology_toolkit.image import dimension_reduce
from mri_histology_toolkit.data_loader import get_leaf_dirs
from scipy.stats import multivariate_normal
from mri_histology_toolkit.thin_plate_spline import warp_image_tps, warp_points_tps
try:
    import tensorflow_addons as tfa
except:
    import numpy as tfa
try:
    import tensorflow as tf
except:
    import numpy as tf

# Import dummy when running in the context of tensorflow container
# replace this workaround
try:
    import nibabel as nib
except:
    import json as nib


def pad_points(points, desired_length):
    padded = np.zeros((desired_length, 2), dtype=np.int32)
    padded[0:min(len(points), desired_length), :] = points[:desired_length, :]
    return padded

def get_files(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def get_all_files(dir):
    all_files = []
    leaf_dirs = get_leaf_dirs(dir)
    for leaf_dir in leaf_dirs:
        all_files += get_files(leaf_dir)
    return all_files

def reverse_center_prostate(img, points, target_size=(512, 512), padding=10, mask_points=None):
    def center_split(total):
        return math.ceil(total / 2), math.floor(total/2)
    
    img = dimension_reduce(img)

    if mask_points is None:
        mask_points = points

    # Trim image
    (ystart, xstart), (ystop, xstop) = mask_points.min(0), mask_points.max(0) + 1
    height, width = (ystop - ystart), (xstop - xstart)
    trimmed_img = img[ystart:ystop, xstart:xstop]

    # Apply padding to square
    if height != width:
        pad_top, pad_bottom = center_split(width - height) if height < width else (0,0)
        pad_left, pad_right = center_split(height - width)if width < height else (0,0)
        trimmed_img = np.pad(trimmed_img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

    # Calculate scale factor
    scale_size = (target_size[0] - (padding * 2), target_size[1] - (padding * 2))
    scale_factor = scale_size[0] / trimmed_img.shape[0]

    # Reverse point operations
    points -= padding
    points = np.rint(points  * (1/scale_factor))

    # Reverse stage 2
    points[:, 0] = points[:, 0] - pad_top
    points[:, 1] = points[:, 1] - pad_left

    # Reverse stage 1
    points[:, 0] = points[:, 0] + ystart
    points[:, 1] = points[:, 1] + xstart

    return points


def center_prostate(img, points, other=None, target_size=(512, 512), padding=10, mask_points=None):
    """
    Takes an input image with a prostate, centers the prostate within the frame, and scales it to fill an
    entire image. Padding is added to this resulting image, and the points are calculated to align with
    original physical structure.
    
    @param img: A numpy array (2 or 3 dimensional) with a masked mri or scan of a prostate
    @param points: A numpy array or list in format [<row, col>, ...] with points on the original image
    @param target_size: The size for the returned image
    @param padding: The amount of padding around the actual prostate in the output image
    
    returns: tuple (img, points) the centered image of size target_size and the same number of dimensions
    as the input. The points recaluclated for the new image
    """
    points = np.array(points)
    assert target_size[0] == target_size[1]
    assert padding >= 0
    
    if mask_points is None:
        mask_points = points

    original_dimensions = len(img.shape)
    img = dimension_reduce(img)
    if other is not None:
        other = dimension_reduce(other)
    
    def center_split(total):
        return math.ceil(total / 2), math.floor(total/2)
    
    # Find bounding box around the nonzero elements of the image
    (ystart, xstart), (ystop, xstop) = mask_points.min(0), mask_points.max(0) + 1
    height, width = (ystop - ystart), (xstop - xstart)
    trimmed_img = img[ystart:ystop, xstart:xstop]
    
    # Adjust points for bounding box cut
    points[:, 0] = points[:, 0] - ystart
    points[:, 1] = points[:, 1] - xstart

    original_y_start, original_y_end, original_x_start, original_x_end = ystart, ystop, xstart, xstop
    # Make bounding box square with padding
    if height != width:
        pad_top, pad_bottom = center_split(width - height) if height < width else (0,0)
        pad_left, pad_right = center_split(height - width)if width < height else (0,0)
        trimmed_img = np.pad(trimmed_img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)
        original_y_start -= pad_top
        original_y_end += pad_bottom
        original_x_start -=pad_left
        original_x_end += pad_right
        points[:, 0] = points[:, 0] + pad_top
        points[:, 1] = points[:, 1] + pad_left
    
    # Resize image to desired output shape and add padding
    scale_size = (target_size[0] - (padding * 2), target_size[1] - (padding * 2))
    scale_factor = scale_size[0] / trimmed_img.shape[0]
    trimmed_img = cv2.resize(trimmed_img, scale_size, interpolation=cv2.INTER_AREA)
    points = np.rint(points  * scale_factor)

    # Scale other
    if other is not None:
        our_padding = int((padding / target_size[0]) * (original_y_end - original_y_start))
        trimmed_other = other[original_y_start-our_padding:original_y_end+our_padding, original_x_start-our_padding:original_x_end+our_padding]
        trimmed_other = cv2.resize(trimmed_other, target_size, interpolation=cv2.INTER_AREA)
    
    # Add final padding
    trimmed_img = np.pad(trimmed_img, padding, 'constant', constant_values=0)
    points += padding
    
    if original_dimensions == 3:
        return trimmed_img[:, :, np.newaxis], points, None if other is None else trimmed_other[:, :, np.newaxis]
    return trimmed_img, points, None if other is None else trimmed_other

def save_image_with_points(img, points, output_file):
    '''
    @param img: A float32 2D or 3D numpy array with values in range [0, 1]
    @param points: A list of points to place on the image (row, col)
    @param output_file: The path to the file to be saved
    '''
    img = dimension_reduce(img)
    points = np.array(points)
    plt.imshow((img * 255).astype(np.uint8), cmap="gray")
    if len(points) > 0:
        plt.scatter(points[:, 1], points[:, 0])
    plt.savefig(output_file)
    plt.clf()


def random_augment(images, points, point_deviation=12):
    points = np.array(points)
    images = np.array(images)
    # batch_size, image dims
    x = np.linspace(16, 496, 8)
    xv, yv = np.meshgrid(x, x, sparse=False)
    coords = np.concatenate((xv.reshape((-1, 1)), yv.reshape((-1, 1))), axis=-1)
    coords = np.repeat(coords[np.newaxis, :, :], len(images), axis=0)
    rand = ((np.random.random_sample(coords.shape) * point_deviation * 2) - point_deviation)
    randomized_grid = coords + rand
    
    warped_image, flow_field = tfa.image.sparse_image_warp(images.astype(np.float32), coords.astype(np.float32), randomized_grid.astype(np.float32))
    
    # Gather new points
    ff = np.array(flow_field)
    starting_coord = np.repeat(np.arange(0, len(images), 1).reshape(-1, 1, 1), points.shape[1], axis=1)
    full_input_point_coords = np.concatenate((starting_coord, points), axis=-1).astype(int).reshape(-1, 3)
    full_input_point_coords[full_input_point_coords < 0] = 0
    full_input_point_coords[full_input_point_coords > 511] = 511
    offsets = np.array([ff[tuple(x)] for x in full_input_point_coords])
    inter_points = points.reshape(-1, 2) + offsets
    inter_points = inter_points.reshape(points.shape)
    
    return warped_image, (inter_points * (points != 0)).astype(int)