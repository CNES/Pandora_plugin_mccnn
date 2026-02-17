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

from typing import Dict, Union, Optional
import os

from json_checker import Checker, And
import xarray as xr
import numpy as np

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
    _SUBPIX = 1
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

        # Select band(s) if multi-band
        selected_band_left = get_band_values(img_left, self._band)
        selected_band_right = get_band_values(img_right, self._band)

        # Disparity range (Pandora allocates disparity planes)
        disparity_range = cost_volume.coords["disp"].data
        disp_min, disp_max = int(disparity_range[0]), int(disparity_range[-1])
        disparity = len(disparity_range)

        # Full canvas (row, col, disparity) initialized with NaN
        row, col = selected_band_left.shape[:2]
        cost_volume_full = np.full((row, col, disparity), np.nan, dtype=np.float32)

        # Expected shrink from configured window size
        window_size = int(self.cfg.get("window_size", self._WINDOW_SIZE))
        n_conv_layer = max(0, (window_size - 1) // 2)

        # Run backend (returns (row_cost_volume, col_cost_volume, disparity)
        # with row_cost_volume=row-2 * n_conv_layer,
        # col_cost_volume=col-2 * n_conv_layer for L conv layers)
        computed_cv = run_mc_cnn_fast(selected_band_left, selected_band_right, disp_min, disp_max, self._model_path)

        # Validate backend output size vs configured window size
        row_cost_volume, col_cost_volume = computed_cv.shape[:2]
        expected_row_cost_volume, expected_col_cost_volume = row - 2 * n_conv_layer, col - 2 * n_conv_layer
        if (row_cost_volume, col_cost_volume) != (expected_row_cost_volume, expected_col_cost_volume):
            raise ValueError(
                f"MC-CNN backend output shape mismatch: got ({row_cost_volume},{col_cost_volume}), "
                f"expected ({expected_row_cost_volume},{expected_col_cost_volume}) "
                f"for window_size={window_size} (L={(window_size - 1)//2}). "
                f"Check that your weights/model_path correspond to the configured window size."
            )

        # Place backend CV centered in the full canvas using the true net offset (which equals L if consistent)
        offset_row = max(0, (row - row_cost_volume) // 2)
        offset_col = max(0, (col - col_cost_volume) // 2)
        cost_volume_full[offset_row : offset_row + row_cost_volume, offset_col : offset_col + col_cost_volume, :] = (
            computed_cv
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


def get_band_values(image_dataset: xr.Dataset, band_name: Optional[str] = None) -> np.ndarray:
    """
    Get values of given band_name from image_dataset as numpy array.

    :param image_dataset: image dataset with band_im coordinate
    :type image_dataset: xr.Dataset
    :param band_name: band name to extract. If None, return all bands values.
    :type band_name: str
    :return: selected band data as numpy array (H, W)
    """
    selection = image_dataset if band_name is None else image_dataset.sel(band_im=band_name)
    return selection["im"].to_numpy()
