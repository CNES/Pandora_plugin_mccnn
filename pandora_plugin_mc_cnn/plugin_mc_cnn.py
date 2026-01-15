#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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

from typing import Dict, Union, Optional
import os
from json_checker import Checker, And
import xarray as xr
import numpy as np

from pandora.img_tools import shift_right_img
from pandora.matching_cost import matching_cost
from pandora.profiler import profile
from mc_cnn.run import run_mc_cnn_fast
from mc_cnn.weights import get_weights


@matching_cost.AbstractMatchingCost.register_subclass("mc_cnn")
class MCCNN(matching_cost.AbstractMatchingCost):
    """

    MCCNN class is a plugin that create a cost volume by calling the McCNN library: a neural network that produce a
    similarity score

    """

    _WINDOW_SIZE = 11
    # _SUBPIX = 1
    # Path to the pretrained model
    _MODEL_PATH = str(get_weights())  # Weights file "mc_cnn_fast_mb_weights.pt" in MC-CNN pip package
    _BAND = None

    @profile("mccnn.__init__")
    def __init__(self, **cfg: Union[int, str]):
        """

        :param cfg: optional configuration, {'matching_cost_method': value,
        'window_size': value, 'subpix': value, 'model_path' :value}
        :type cfg: dictionary
        """
        super().instantiate_class(**cfg)
        self._model_path = str(self.cfg["model_path"])

    def check_conf(self, **cfg: Union[int, str]) -> Dict[str, Union[int, str]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: matching_cost configuration
        :type cfg: dict
        :return cfg: matching_cost configuration updated
        :rtype: dict
        """
        cfg = super().check_conf(**cfg)

        if "model_path" not in cfg:
            cfg["model_path"] = self._MODEL_PATH

        schema = self.schema
        schema["matching_cost_method"] = And(str, lambda input: "mc_cnn")
        schema["window_size"] = And(int, lambda x: x == 11)
        schema["model_path"] = And(str, lambda x: os.path.exists(x))

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    @profile("mccnn.compute_cost_volume")
    def compute_cost_volume(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cost_volume: xr.Dataset,
    ) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images

        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :param cost_volume: an empty cost volume
        :type cost_volume: xr.Dataset
        :return: the cost volume dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
        :rtype:
            xarray.Dataset
        """
        # check band parameter
        self.check_band_input_mc(img_left, img_right)

        # Contains the shifted right images
        imgs_right_shift = shift_right_img(img_right, self._subpix, self._band, self._spline_order)

        # If multiband, select the corresponding band
        if self._band is None:
            img_left_np = img_left["im"].data
            imgs_right_shift_np = [img["im"].data for img in imgs_right_shift]
        else:
            band_index = list(img_left.band_im.data).index(self._band)
            img_left_np = img_left["im"].data[band_index, :, :]
            imgs_right_shift_np = []
            for img in imgs_right_shift:
                if "band_im" in img:
                    imgs_right_shift_np.append(img["im"].data[band_index, :, :])
                else:
                    imgs_right_shift_np.append(img["im"].data)


        # selected_band_left = get_band_values(img_left, self._band)
        # selected_band_right = get_band_values(img_right, self._band)

        disparity_range = cost_volume.coords["disp"].data
        disp_min, disp_max = disparity_range[0], disparity_range[-1]
        offset_row_col = cost_volume.attrs["offset_row_col"]

        cv = np.full(
            (img_left_np.shape[0], img_left_np.shape[1], len(disparity_range)), np.nan, dtype=np.float32
        )

        # If offset, do not consider border position for cost computation
        for idx_right in range(self._subpix):
            img_right_np = imgs_right_shift_np[idx_right]

            if idx_right > 0:
                img_right_np = np.concatenate((img_right_np, np.full((img_right_np.shape[0], 1), np.nan)),
                                              axis=1)
            
            if offset_row_col != 0:
                cv[offset_row_col:-offset_row_col, offset_row_col:-offset_row_col, idx_right::self._subpix] = run_mc_cnn_fast(
                    img_left_np.astype(np.float32), img_right_np.astype(np.float32),
                    disp_min, disp_max, self._model_path
                )
            else:
                cv[:, :, idx_right::self._subpix] = run_mc_cnn_fast(
                    img_left_np.astype(np.float32), img_right_np.astype(np.float32),
                    disp_min, disp_max, self._model_path)

            # For subpix step, remove latest disparity (replaced by a column of NaNs)
            if idx_right == 0:
                disp_max -= 1

        index_col = cost_volume.attrs["col_to_compute"]
        index_col = index_col - img_left.coords["col"].data[0]  # If first col coordinate is not 0
        cost_volume["cost_volume"].data = cv[:, index_col, :]
        # Allocate the xarray cost volume
        cost_volume.attrs.update(
            {
                "type_measure": "min",
                "cmax": 1,
            }
        )

        return cost_volume


def get_band_values(image_dataset: xr.Dataset, band_name: Optional[str] = None) -> np.ndarray:
    """
     Get values of given band_name from image_dataset as numpy array.

    :param image_dataset: image dataset
    :type image_dataset: xr.Dataset with band_im coordinate
    :param band_name: nome of the band to extract. If None (default) return all bands values.
    :type band_name: str
    :return: selected band
    :rtype: np.ndarray
    """
    selection = image_dataset if band_name is None else image_dataset.sel(band_im=band_name)
    return selection["im"].to_numpy()
