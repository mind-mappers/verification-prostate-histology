#!/usr/bin/python
# -*- coding: utf-8 -*-
from homologous_point_prediction.data_processing.helpers import random_augment, save_image_with_points, pad_points
from mri_histology_toolkit.data_loader import get_child_dirs, get_leaf_dirs
from mri_histology_toolkit.image import dimension_reduce
from homologous_point_prediction.data_processing.augmentation import rotate, flip
import tensorflow as tf
import numpy as np
import random
import json
import math
import os

def extract_normal(slide_dir, use_masked, pair_rate=0.0, edge_rate=0.5):

    # Determine the random state of this slide
    use_pair = random.random() < pair_rate
    use_edge = random.random() < edge_rate
    mri_point_filename = 'mri_points.npy' if not use_edge else 'edge_mri_points.npy'
    hist_point_filename = 'hist_points.npy' if not use_edge else 'edge_hist_points.npy'
    pair_mod = random.randint(0, 1)

    mod_1 = 0 if not use_pair else pair_mod
    mod_2 = 1 if not use_pair else pair_mod

    if use_pair:
        if pair_mod == 0:
            fp = np.load(os.path.join(slide_dir, hist_point_filename))
            fi = np.load(os.path.join(slide_dir, 'fixed.npy'))
            mp = fp
            mi = fi
        else:
            mp = np.load(os.path.join(slide_dir, mri_point_filename))
            mi = np.load(os.path.join(slide_dir, ('moving.npy' if not use_masked else 'masked_moving.npy')))
            fp = mp
            fi = mi
    else:
        fp = np.load(os.path.join(slide_dir, hist_point_filename))
        fi = np.load(os.path.join(slide_dir, 'fixed.npy'))
        mp = np.load(os.path.join(slide_dir, mri_point_filename))
        mi = np.load(os.path.join(slide_dir, ('moving.npy' if not use_masked else 'masked_moving.npy')))

    return (fi, mi, fp, mp, mod_1, mod_2)

def filter_points(fp, mp):
    mp = mp * (mp[:, :1] >= 0) *  (mp[:, :1] < 512)
    mp = mp * (mp[:, 1:] >= 0) *  (mp[:, 1:] < 512)
    fp = fp * (mp[:, :1] >= 0) *  (mp[:, :1] < 512)
    fp = fp * (mp[:, 1:] >= 0) *  (mp[:, 1:] < 512)

    mp = mp * (fp[:, :1] >= 0) *  (fp[:, :1] < 512)
    mp = mp * (fp[:, 1:] >= 0) *  (fp[:, 1:] < 512)
    fp = fp * (fp[:, :1] >= 0) *  (fp[:, :1] < 512)
    fp = fp * (fp[:, 1:] >= 0) *  (fp[:, 1:] < 512)
    return fp, mp


def extract_slide(slide_dir, num_points, use_masked=True, rotation_range=[0,0], pair_rate=0.0, max_points=1, edge_rate=0.5):
    fi, mi, fp, mp, mod_1, mod_2 = extract_normal(slide_dir, use_masked, pair_rate=pair_rate, edge_rate=edge_rate)
    random_point_indices = np.random.choice(len(fp), min(num_points, len(fp)), replace=False)#[:max_points]
    fp, mp = fp[random_point_indices], mp[random_point_indices]

    # Augmentation Random
    rotation_angle = random.randint(rotation_range[0], rotation_range[1])
    tps_augment_max_a = 12 + random.randint(0, 4)
    tps_augment_max_b = 12 + random.randint(0, 4)#random.randint(0, 12)

    # Rotate the moving image if needed
    if rotation_angle != 0:
        (mi, mp) = rotate(mi, rotation_angle, mp)
    
    if tps_augment_max_a != 0:
        fi, fp = random_augment(fi.reshape((1, 512, 512, 1)), fp.reshape((1, -1, 2)), tps_augment_max_a)
        fi, fp = fi[0], fp[0]

    if tps_augment_max_b != 0:
        mi, mp = random_augment(mi.reshape((1, 512, 512, 1)), mp.reshape((1, -1, 2)), tps_augment_max_b)
        mi, mp = mi[0], mp[0]

    fp, mp = pad_points(fp, num_points), pad_points(mp,  num_points)
    fp, mp = filter_points(fp, mp)

    return (fi, mi, fp, mp, mod_1, mod_2)


class MultiPointDataLoader(tf.keras.utils.Sequence):

    """Lazy loading for memory use min. Loading all images first will improve performance"""

    def __init__(self, config_path, batch_size=16, num_points=16, warped_pair_rate=0.0, edge_rate=0.5):
        self.batch_size = batch_size
        self.num_points = num_points
        self._parse_config(config_path)
        self.slide_dirs = []
        self.warped_pair_rate = warped_pair_rate
        self.edge_rate = edge_rate

        patients = get_child_dirs(self.parent_dir)
        for include_patient in self.include_patients:
            if include_patient not in patients:
                print('Warning: patient {0} not found...ignoring'.format(include_patient))
            else:
                self.slide_dirs += get_child_dirs(os.path.join(self.parent_dir, include_patient), full_path=True)
        self.indices = np.arange(len(self.slide_dirs))
        self.slide_dirs = np.array(self.slide_dirs)
        np.random.shuffle(self.indices)

    def _parse_config(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        self.parent_dir = config['parent_dir']
        self.include_patients = config['include_patients']
        self.augment_rotation = config['augment_rotation_range']
        self.use_masked = config['use_masked']
        self.use_warped_pairs = config['use_warped_pairs'] 
        assert len(self.parent_dir) > 0
        assert len(self.include_patients) > 0
        assert len(self.augment_rotation) == 2

    def __len__(self):
        return int(math.floor(len(self.slide_dirs) / self.batch_size))

    def __getitem__(self, batch_index):
        batch_indices = self.indices[batch_index
            * self.batch_size:(batch_index + 1) * self.batch_size]
        batch_sample_dirs = self.slide_dirs[batch_indices]
        return self.extract_data(batch_sample_dirs)

    def extract_data(self, dirs):
        (fixed_images, moving_images, fixed_points, moving_points, mods_1, mods_2) = [], [], [], [], [], []
        for sample_dir in dirs:
            (fi, mi, fp, mp, mod_1, mod_2) = extract_slide(sample_dir, self.num_points, use_masked=self.use_masked, rotation_range=self.augment_rotation, pair_rate=self.warped_pair_rate, edge_rate=self.edge_rate)
            fixed_images.append(fi)
            moving_images.append(mi)
            fixed_points.append(fp)
            moving_points.append(mp)
            mods_1.append(mod_1)
            mods_2.append(mod_2)
        return [np.array(fixed_images), np.array(moving_images), np.array(fixed_points), np.array(mods_1), np.array(mods_2)], np.array(moving_points)

    def save_sample_batch(self, output_dir):
        (X, y) = self[0]
        (fixed_images, moving_images, fixed_points, mod1, mod2) = X
        for i in range(min(6, len(y))):
            save_image_with_points(dimension_reduce(fixed_images[i]), fixed_points[i], os.path.join(output_dir, '{0} _data_loader_sample_fixed_with_point.png'.format(i)))
            save_image_with_points(dimension_reduce(moving_images[i]), y[i], os.path.join(output_dir, '{0}_data_loader_sample_moving_with_point.png'.format(i)))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def summarize(self):
        arrow = ''.join(['-'] * 75) + '>'
        print('Data Loading Summary')
        print('Num Points per image', arrow, self.num_points)
        print('Batch Size', arrow, self.batch_size)
        print('Rotation Min', arrow, self.augment_rotation[0])
        print('Rotation Max', arrow, self.augment_rotation[1])


        # print("Use Masked", arrow, self.use_masked)
