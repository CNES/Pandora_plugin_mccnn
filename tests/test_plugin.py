#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2019 Centre National d'Etudes Spatiales
"""
This module contains functions to test Pandora + plugin_mc-cnn
"""

from tempfile import TemporaryDirectory
import unittest
import rasterio
import numpy as np
import xarray as xr

import pandora
from plugin_mc_cnn.plugin_mc_cnn import MCCNN


class TestPlugin(unittest.TestCase):
    """
    TestPlugin class allows to test Pandora + plugin_mc-cnn
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        self.disp_ref = rasterio.open('tests/image/disp_ref.tif').read(1)
        self.disp_sec = rasterio.open('tests/image/disp_sec.tif').read(1)

    def error(self, data, gt, threshold, unknown_disparity=0):
        """
        Percentage of bad pixels whose error is > threshold

        """
        row, col = data.shape
        nb_error = 0
        for r in range(row):
            for c in range(col):
                if gt[r, c] != unknown_disparity:
                    if abs((data[r, c] + gt[r, c])) > threshold:
                        nb_error += 1

        return nb_error / float(row * col)

    def test_mc_cnn(self):
        """"
        Test Pandora + plugin_mc-cnn

        """
        # Create temporary directory
        with TemporaryDirectory() as tmp_dir:
            pandora.main('tests/test_cfg_accurate.json', tmp_dir, verbose=False)

            # Check the reference disparity map
            if self.error(rasterio.open(tmp_dir + '/ref_disparity.tif').read(1), self.disp_ref, 1) > 0.17:
                raise AssertionError

            # Check the secondary disparity map
            if self.error(-1 * rasterio.open(tmp_dir + '/sec_disparity.tif').read(1), self.disp_sec, 1) > 0.17:
                raise AssertionError

    def test_invalidates_cost(self):
        """
        Test invalidates_cost function : this function mask invalid pixels

        """
        # ------------ Test the method with a reference mask ( secondary mask contains valid pixels ) ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.zeros((13, 13), dtype=np.float64)

        mask = np.array(([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.zeros((13, 13), dtype=np.float64)

        # Secondary mask contains valid pixels
        mask = np.zeros((13, 13), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        # Cost volume before invalidation, disparities = -1, 0, 1
        cv_before_invali = np.array([[[np.nan, 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., np.nan]],

                                     [[np.nan, 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., np.nan]],

                                     [[np.nan, 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., np.nan]]], dtype=np.float32)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan, np.nan, np.nan],
                                      [0., 0., 0.],
                                      [0., 0., np.nan]],

                                     [[np.nan, 0., 0.],
                                      [np.nan, np.nan, np.nan],
                                      [np.nan, np.nan, np.nan]],

                                     [[np.nan, 0., 0.],
                                      [np.nan, np.nan, np.nan],
                                      [np.nan, np.nan, np.nan]]], dtype=np.float32)

        stereo_ = MCCNN(**{'stereo_method': 'mc_cnn', 'window_size': 11, 'subpix': 1, 'mc_cnn_arch': 'fast',
                           'model_path': 'weights/mc_cnn_fast_mb_weights.pt'})

        cv = stereo_.invalidates_cost(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1, cv=cv_before_invali,
                                      **{'valid_pixels': 0, 'no_data': 1})
        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv, cv_ground_truth)

        # ------------ Test the method with a secondary mask ( reference mask contains valid pixels ) ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.zeros((13, 13), dtype=np.float64)

        # Reference mask contains valid pixels
        mask = np.zeros((13, 13), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.zeros((13, 13), dtype=np.float64)

        mask = np.array(([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), dtype=np.int16)

        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        # Cost volume before invalidation, disparities = -1, 0, 1
        cv_before_invali = np.array([[[np.nan, 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., np.nan]],

                                     [[np.nan, 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., np.nan]],

                                     [[np.nan, 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., np.nan]]], dtype=np.float32)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan,  np.nan,   0.],
                                      [np.nan,     0.,   0.],
                                      [0.,         0., np.nan]],

                                     [[np.nan,     0., np.nan],
                                      [0.,     np.nan, np.nan],
                                      [np.nan, np.nan, np.nan]],

                                     [[np.nan,     0., np.nan],
                                      [0.,     np.nan, np.nan],
                                      [np.nan, np.nan, np.nan]]], dtype=np.float32)

        stereo_ = MCCNN(**{'stereo_method': 'mc_cnn', 'window_size': 11, 'subpix': 1, 'mc_cnn_arch': 'fast',
                           'model_path': 'weights/mc_cnn_fast_mb_weights.pt'})

        cv = stereo_.invalidates_cost(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1, cv=cv_before_invali,
                                      **{'valid_pixels': 0, 'no_data': 1})

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv, cv_ground_truth)


if __name__ == '__main__':
    unittest.main()
