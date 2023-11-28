import os
import cv2
import numpy as np
import math

def flip(img, point=None):
    '''
    @param img: Image to be flipped (horizontally)
    @param point: Point on the image to flip along with image

    returns: An image flipped horizontally and a point flipped the same way
    '''
    orig_shape = img.shape
    flipped = cv2.flip(img, 1).reshape(orig_shape)
    if type(point) != type(None):
        image_size = img.shape[:2]
        image_max_indices = (image_size[0] - 1, image_size[1] - 1)
        point[1] = (image_max_indices[1] - point[1])
        return flipped, point
    return flipped


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    oy, ox  = origin
    py, px  = point
    angle = angle * -1

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qy, qx]

def rotate(img, angle, point=None):
    '''
    @param img: Image to be rotated
    @param angle: Angle of rotation in degrees
    @param point: Point on the image to rotate along with image <row, col> (can be (,2) or (num_points, 2))

    returns: An image rotated by angle
             <row, col> point translated to compenstate for augmentation if input point provided
    '''
    if type(point) != type(None):
        # Standardize point(s) shape
        point = np.array(point)
        original_point_shape = point.shape
        point = point.reshape((-1, 2))

        # Calculate image data needed to rotate point(s)
        rads = math.radians(angle)
        oy, ox = tuple(np.array(img.shape[1::-1]) / 2)

        # Rotate the point(s)
        point = np.array([rotate_point((oy, ox), point[i, :], rads) for i in range(len(point))])

        # restore original shape
        point = point.reshape(original_point_shape)
        point = np.rint(point)

    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result[:, :, np.newaxis] if type(point) == type(None) else (result[:, :, np.newaxis], point)
