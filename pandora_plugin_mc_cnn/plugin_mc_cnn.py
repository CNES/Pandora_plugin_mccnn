#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of Pandora plugin MC-CNN
#
#     https://github.com/CNES/Pandora_plugin_mccnn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains all functions to calculate the cost volume with mc-cnn networks
"""

from typing import Dict, Union
import os
from json_checker import Checker, And
import xarray as xr
import numpy as np

from pandora.matching_cost import matching_cost
from mc_cnn.run import run_mc_cnn_fast
from mc_cnn.weights import get_weights


@matching_cost.AbstractMatchingCost.register_subclass("mc_cnn")
class MCCNN(matching_cost.AbstractMatchingCost):
    """

    MCCNN class is a plugin that create a cost volume by calling the McCNN library: a neural network that produce a
    similarity score

    """

    _WINDOW_SIZE = 11
    _SUBPIX = 1
    # Path to the pretrained model
    _MODEL_PATH = str(get_weights())  # Weights file "mc_cnn_fast_mb_weights.pt" in MC-CNN pip package

    def __init__(self, **cfg: Union[int, str]):
        """

        :param cfg: optional configuration, {'matching_cost_method': value,
        'window_size': value, 'subpix': value, 'model_path' :value}
        :type cfg: dictionary
        """
        self.cfg = self.check_config(**cfg)
        self._model_path = str(self.cfg["model_path"])
        self._window_size = self.cfg["window_size"]
        self._subpix = self.cfg["subpix"]

    def check_config(self, **cfg: Union[int, str]) -> Dict[str, Union[int, str]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: matching_cost configuration
        :type cfg: dict
        :return cfg: matching_cost configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE
        if "subpix" not in cfg:
            cfg["subpix"] = self._SUBPIX
        if "model_path" not in cfg:
            cfg["model_path"] = self._MODEL_PATH

        schema = {
            "matching_cost_method": And(str, lambda x: x == "mc_cnn"),
            "window_size": And(int, lambda x: x == 11),
            "subpix": And(int, lambda x: x == 1),
            "model_path": And(str, lambda x: os.path.exists(x)),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the optimization method

        """
        print("MC-CNN similarity measure")

    def compute_cost_volume(
        self, img_left: xr.Dataset, img_right: xr.Dataset, disp_min: int, disp_max: int
    ) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images

        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :return: the cost volume dataset
        :rtype:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        """

        # Disparity range
        if self._subpix == 1:
            disparity_range = np.arange(disp_min, disp_max + 1)
        else:
            disparity_range = np.arange(disp_min, disp_max, step=1 / float(self._subpix))
            disparity_range = np.append(disparity_range, [disp_max])

        offset_row_col = int((self._window_size - 1) / 2)
        cv = np.zeros((img_right["im"].shape[0], img_left["im"].shape[1], len(disparity_range)), dtype=np.float32)
        cv += np.nan

        # If offset, do not consider border position for cost computation
        if offset_row_col != 0:
            cv[offset_row_col:-offset_row_col, offset_row_col:-offset_row_col, :] = run_mc_cnn_fast(
                img_left, img_right, disp_min, disp_max, self._model_path
            )
        else:
            cv = run_mc_cnn_fast(img_left, img_right, disp_min, disp_max, self._model_path)

        # Allocate the xarray cost volume
        metadata = {
            "measure": "mc_cnn_fast",
            "subpixel": self._subpix,
            "offset_row_col": int((self._window_size - 1) / 2),
            "window_size": self._window_size,
            "type_measure": "min",
            "cmax": 1,
        }

        cv = self.allocate_costvolume(img_left, self._subpix, disp_min, disp_max, self._window_size, metadata, cv)

        return cv
