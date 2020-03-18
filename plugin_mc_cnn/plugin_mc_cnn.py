#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2019 Centre National d'Etudes Spatiales

"""
This module contains all functions to calculate the cost volume with mc-cnn networks
"""

import logging
import numpy as np

from pandora.stereo import stereo
from mc_cnn.run import run_mc_cnn_fast, run_mc_cnn_accurate


@stereo.AbstractStereo.register_subclass('mc_cnn')
class MCCNN(stereo.AbstractStereo):
    """

    MCCNN class is a plugin that create a cost volume by calling the McCNN library: a neural network that produce a
    similarity score

    """
    _architecture = 'fast'
    _window_size = 11
    _subpix = 1
    _trained_net = None

    def __init__(self, **cfg):
        """

        :param cfg: optional configuration, {'stereo_method': value, 'mc_cnn_arch': 'fast' | 'accurate' }
        :type cfg: dictionary
        """
        self.check_config(**cfg)

    def check_config(self, **cfg):
        """
        Check and update the configuration

        :param cfg: configuration
        :type cfg: dictionary
        """
        if 'mc_cnn_arch' in cfg:
            if cfg['mc_cnn_arch'] == 'fast':
                self._architecture = 'fast'
            elif cfg['mc_cnn_arch'] == 'accurate':
                self._architecture = 'accurate'
            else:
                logging.error("No mc-cnn architecture name {} supported".format(cfg['mc_cnn_arch']))
                exit()

        if 'window_size' in cfg:
            if int(cfg['window_size']) != 11:
                logging.error("Mc-cnn similarity measure only accepts window_size = 11")
                exit()

        if 'subpix' in cfg:
            if int(cfg['subpix']) != 1:
                logging.error("Mc-cnn similarity measure only accepts subpixel = 1")
                exit()

        if not 'model_path' in cfg:
            logging.error("A path to a trained network is required")
            exit()
        else:
            self._trained_net = cfg['model_path']

    def desc(self):
        """
        Describes the optimization method

        """
        print('MC-CNN similarity measure')

    def compute_cost_volume(self, img_ref, img_sec, disp_min, disp_max, **cfg):
        """
        Computes the cost volume for a pair of images

        :param img_ref: reference Dataset image
        :type img_ref:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :param cfg: images configuration containing the mask convention : valid_pixels, no_data
        :type cfg: dict
        :return: the cost volume
        :rtype: xarray.Dataset, with the data variables cost_volume 3D xarray.DataArray (row, col, disp)
        """
        if self._architecture == 'fast':
            cv = run_mc_cnn_fast(img_ref, img_sec, disp_min, disp_max, self._trained_net)

        # Accurate architecture
        else:
            cv = run_mc_cnn_accurate(img_ref, img_sec, disp_min, disp_max, self._trained_net)

        # Invalid invalid pixels : cost of invalid pixels will be = np.nan
        offset = int((self._window_size - 1) / 2)
        mask_ref, mask_sec = self.masks_dilatation(img_ref, img_sec, offset, self._window_size, self._subpix, cfg)
        ny_, nx_ = mask_ref.shape
        disparity_range = np.arange(disp_min, disp_max + 1)

        for disp in disparity_range:
            # range in the reference image
            p = (max(0 - disp, 0), min(nx_ - disp, nx_))
            # range in the secondary image
            q = (max(0 + disp, 0), min(nx_ + disp, nx_))
            d = int(disp - disp_min)

            cv[:, p[0]:p[1], d] += mask_sec[0].data[:, q[0]:q[1]] + mask_ref.data[:, p[0]:p[1]]

        # Allocate the xarray cost volume
        metadata = {"measure": 'mc_cnn_' + self._architecture, "subpixel": self._subpix,
                    "offset_row_col": int((self._window_size - 1) / 2), "window_size": self._window_size,
                    "type_measure": "min", "cmax": 1}
        cv = self.allocate_costvolume(img_ref, self._subpix, disp_min, disp_max, self._window_size, metadata, cv)

        return cv
