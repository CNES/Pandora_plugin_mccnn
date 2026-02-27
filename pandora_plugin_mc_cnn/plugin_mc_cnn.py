# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to mc-cnn method used in the cost volume measure step.
"""

from typing import Dict, Union
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
    MC-CNN plugin that computes a cost volume using MC-CNN fast features and a cosine similarity loop.

    CPU-only baseline (even if a GPU is present).
    """

    _WINDOW_SIZE = 11
    _MODEL_PATH = str(get_weights())  # Pretrained weights from mc_cnn package
    _BAND = None

    @profile("mc_cnn.__init__")
    def __init__(self, **cfg: Union[int, str]):
        """
        :param cfg: {
            'matching_cost_method': str,
            'window_size': int,
            'subpix': int,
            'model_path': str,
        }
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

        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE

        schema = self.schema
        schema["matching_cost_method"] = And(str, lambda x: x == "mc_cnn")
        schema["window_size"] = And(int, lambda x: x == 11)
        schema["model_path"] = And(str, lambda x: os.path.exists(x))

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    @profile("mc_cnn.compute_cost_volume")
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
        # Check band parameter
        self.check_band_input_mc(img_left, img_right)

        # Contains the shifted right images
        imgs_right_shift = shift_right_img(img_right, self._subpix, self._band, self._spline_order)

        # Select band(s) if multi-band
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

        # Disparity range (Pandora allocates disparity planes)
        disparity_range = cost_volume.coords["disp"].data
        disp_min, disp_max = int(disparity_range[0]), int(disparity_range[-1])
        disparity = len(disparity_range)
        offset_row_col = cost_volume.attrs["offset_row_col"]

        # Full canvas (row, col, disparity) initialized with NaN
        row, col = img_left_np.shape[:2]
        cost_volume_full = np.full((row, col, disparity), np.nan, dtype=np.float32)

        # Expected shrink from configured window size
        window_size = int(self.cfg.get("window_size", self._WINDOW_SIZE))
        n_conv_layer = max(0, (window_size - 1) // 2)

        # Run backend (returns (row_cost_volume, col_cost_volume, disparity)
        # with row_cost_volume=row-2 * n_conv_layer,
        # col_cost_volume=col-2 * n_conv_layer for L conv layers)
        for idx_right in range(self._subpix):
            img_right_np = imgs_right_shift_np[idx_right]

            if idx_right > 0:
                # If the image has been shifted, the last line is removed,
                # which is not compatible with the network's input shape.
                # In that case, we add a column of NaN (with a value).
                # The value is not NaN due to issues with the network.
                img_right_np = np.concatenate((img_right_np, np.full((img_right_np.shape[0], 1), 0)), axis=1)

            computed_cost_volume = run_mc_cnn_fast(
                img_left_np.astype(np.float32),
                img_right_np.astype(np.float32),
                disp_min,
                disp_max,
                self._model_path,
            )
            if offset_row_col != 0:
                cost_volume_full[
                    offset_row_col:-offset_row_col, offset_row_col:-offset_row_col, idx_right :: self._subpix
                ] = computed_cost_volume
            else:
                cost_volume_full[:, :, idx_right :: self._subpix] = computed_cost_volume

            # For subpix step, remove latest disparity (replaced by a column of NaNs)
            if idx_right == 0:
                disp_max -= 1

        # Validate backend output size vs configured window size
        row_cost_volume, col_cost_volume = computed_cost_volume.shape[:2]
        expected_row_cost_volume, expected_col_cost_volume = row - 2 * n_conv_layer, col - 2 * n_conv_layer
        if (row_cost_volume, col_cost_volume) != (expected_row_cost_volume, expected_col_cost_volume):
            raise ValueError(
                f"MC-CNN backend output shape mismatch: got ({row_cost_volume},{col_cost_volume}), "
                f"expected ({expected_row_cost_volume},{expected_col_cost_volume}) "
                f"for window_size={window_size} (L={(window_size - 1)//2}). "
                f"Check that your weights/model_path correspond to the configured window size."
            )

        # Select requested columns
        index_col = np.asarray(cost_volume.attrs["col_to_compute"])
        # Rebase if first column coordinate != 0
        index_col = index_col - img_left.coords["col"].data[0]

        # Fill Pandora cost volume slice (row, col_to_compute, disp)
        cost_volume["cost_volume"].data = cost_volume_full[:, index_col, :]

        # Metadata
        cost_volume.attrs.update(
            {
                "type_measure": "min",
                "cmax": 1,
            }
        )

        return cost_volume
