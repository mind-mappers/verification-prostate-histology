from mri_histology_toolkit.data_loader import load_mri, load_histology, get_child_dirs
from helpers import save_image_with_points, center_prostate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import cv2
import json
import os
import sys


def get_slide_dirs(include, blacklist, parent_dir):
    """
    param @include: List of tuples <patient, slide> to include
    param @blacklist: List of tuples <patient, slide> to blacklist
    """
    to_use_slides = [] # list fo tuples <patient, slide>

    if len(include) > 0:
        for to_include in include:
            patient, slide = to_include
            if slide == "*":
                all_slides = get_child_dirs(os.path.join(parent_dir, patient))
                to_use_slides += [(patient, s) for s in all_slides]
            else:
                to_use_slides.append((patient, slide))
    else:
        for patient in get_child_dirs(parent_dir):
            slides = get_child_dirs(os.path.join(parent_dir, patient))
            to_use_slides += [(patient, s) for s in slides]
        
    # Filter out all of the black list slides
    blacklisted_patients = [x[0] for x in blacklist if x[1] == "*" ]
    id_strings = ["{0}_{1}".format(x[0], x[1]) for x in blacklist]
    to_use_slides = [x for x in to_use_slides if x[0] not in blacklisted_patients and "{0}_{1}".format(x[0], x[1]) not in id_strings]
    return to_use_slides


def handle_slide(mri_image_path, hist_image_path, points_path, output_dir):
    try:
        mri_image = load_mri(mri_image_path)
        hist_image = load_histology(hist_image_path)
        points = pd.read_csv(points_path, header=None).to_numpy().astype(int)
    except Exception as e:
        print(e)
        print("Failed to load one of the following:\n{0}\n{1}\n{2}".format(mri_image_path, hist_image_path, points_path))
        print("Skipping the slide\n")
        return
    hist_points = np.flip(points[:, :2], axis=-1)
    mri_points = np.flip(points[:, 2:], axis=-1) # We flip the columns from (x, y) to (y, x) coords

    # Rescale the values
    _, mri_points, mri_image = center_prostate(mri_image, mri_points, other=mri_image, padding=50)
    hist_image, hist_points, _ = center_prostate(hist_image, hist_points, padding=50)

    # Save 
    #os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_dir):
        #np.save(os.path.join(output_dir, "fixed.npy"), hist_image[:, :, np.newaxis])
        #np.save(os.path.join(output_dir, "moving.npy"), mri_image[:, :, np.newaxis])
        # Save Points
        np.save(os.path.join(output_dir, "edge_hist_points.npy"), hist_points)
        np.save(os.path.join(output_dir, "edge_mri_points.npy"), mri_points)
        # Human readable images
        save_image_with_points(hist_image, hist_points, os.path.join(output_dir, "edge_hist_fixed.png"))
        save_image_with_points(mri_image, mri_points, os.path.join(output_dir, "edge_unmasked_mri_moving.png"))


def process_all(process_json_file):
    with open(process_json_file) as f:
        config = json.load(f)
    
    include_list = config["sample_include"]
    black_list = config["sample_blacklist"]
    parent_dir = config["parent_dir"]
    output_dir = config["output_parent_dir"]
    mri_image_file = config["mri_image_filename"]
    hist_image_file = config["hist_image_filename"]
    hist_image_fallback_filename = config["hist_image_fallback_filename"]
    points_file = config["csv_filename"]

    to_use_slides = get_slide_dirs(include_list, black_list, parent_dir)

    for patient, slide in to_use_slides:
        input_slide_dir = os.path.join(parent_dir, patient, slide)
        output_slide_dir = os.path.join(output_dir, patient, slide)
        mri_image_path = os.path.join(input_slide_dir, mri_image_file)
        hist_image_path = os.path.join(input_slide_dir, hist_image_file) if os.path.exists(os.path.join(input_slide_dir, hist_image_file)) else os.path.join(input_slide_dir, hist_image_fallback_filename)
        points_csv_path = os.path.join(input_slide_dir, points_file)
        try:
            handle_slide(mri_image_path, hist_image_path, points_csv_path, output_slide_dir)
        except:
            print("SKIPPING SLIDE +++++++ error")



# Run Script
if len(sys.argv) < 2:
    print("Please provide the conofig json path as an argument")
else:
    process_all(sys.argv[1])