from mri_histology_toolkit.thin_plate_spline import warp_image_tps, warp_points_tps
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os


def warp_image(image, input_points, output_points):
    '''
    Warps an image based on the input and output points
    '''
    input_points = np.flip(input_points, axis=-1)
    output_points = np.flip(output_points, axis=-1)
    return warp_image_tps(image.astype(np.float32), input_points.astype(np.float32), output_points.astype(np.float32), [512, 512], normalize_points=True)

def warp_points(points, input_points, output_points):
    '''
    Warps points according to a tps resolved from input and output points
    '''
    return warp_points_tps(points, input_points, output_points, (512, 512), (512, 512), normalize_points=True)

