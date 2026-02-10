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

from typing import Dict, Union, Optional
import os
import json
import time
import threading
from pathlib import Path

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
    _SUBPIX = 1
    _MODEL_PATH = str(get_weights())  # Pretrained weights from mc_cnn package
    _BAND = None
    _FRAMEWORK = "pytorch"  # Default framework (CPU-only)
    _VARIANT = "baseline"   # Default variant
    _PROVIDER = "cpu_base"

    @profile("mc_cnn.__init__")
    def __init__(self, **cfg: Union[int, str]):
        """
        :param cfg: {
            'matching_cost_method': str,
            'window_size': int,
            'subpix': int,
            'model_path': str,
            'framework': 'pytorch'|'onnx'|'openvino',
            'variant': 'baseline'|'opt1'|'opt2'|'cpp'|'cpp2'|'opt1_notorch'|'opt2_notorch'|'cpp_notorch'|'cpp2_notorch',
            'provider': 'cpu_base'|'openvino',
            'model_name': str (optional; used to select a specific ONNX/IR file)
        }
        """
        super().instantiate_class(**cfg)
        self._model_path = str(self.cfg["model_path"])
        self._framework = str(self.cfg.get("framework", self._FRAMEWORK))
        self._variant = str(self.cfg.get("variant", self._VARIANT))
        self._provider = str(self.cfg.get("provider", self._PROVIDER))
        # keep the configured window size (Pandora is the source of truth)
        self._window_size = int(self.cfg.get("window_size", self._WINDOW_SIZE))

    def check_conf(self, **cfg: Union[int, str]) -> Dict[str, Union[int, str]]:
        """
        Add defaults and validate configuration.
        window_size is required (and should match the weights you provide).
        """
        cfg = super().check_conf(**cfg)

        if "model_path" not in cfg:
            cfg["model_path"] = self._MODEL_PATH

        if "framework" not in cfg:
            cfg["framework"] = self._FRAMEWORK

        if "variant" not in cfg:
            cfg["variant"] = self._VARIANT

        if "provider" not in cfg:
            cfg["provider"] = self._PROVIDER

        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE  # default if not provided

        schema = self.schema
        schema["matching_cost_method"] = And(str, lambda x: x == "mc_cnn")
        schema["window_size"] = And(int, lambda x: x in [7, 11, 13, 15])
        schema["model_path"] = And(str, lambda x: os.path.exists(x))
        schema["framework"] = And(str, lambda x: x in ["pytorch", "onnx", "openvino"])
        schema["variant"] = And(str, lambda x: x in ["baseline", "opt1", "opt1_notorch", "opt2", "opt2_notorch", "cpp", "cpp2", "cpp_notorch", "cpp2_notorch"])
        schema["provider"] = And(str, lambda x: x in ["cpu_base", "openvino"])
        if "model_name" in cfg:
            schema["model_name"] = str

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
        Compute cost volume for a pair of images using MC-CNN fast features (CPU-only).
        Emits PROFILING_TOTAL_CV marker with total time/memory.
        Writes metrics_total.json with total_cv_time and total_cv_mem (true peak within this function).
        """
        # Check band parameter
        self.check_band_input_mc(img_left, img_right)

        # Select band(s) if multi-band
        selected_band_left = get_band_values(img_left, self._band)
        selected_band_right = get_band_values(img_right, self._band)

        # Disparity range (Pandora allocates D planes)
        disparity_range = cost_volume.coords["disp"].data
        disp_min, disp_max = int(disparity_range[0]), int(disparity_range[-1])
        D = len(disparity_range)

        # Full canvas (H, W, D) initialized with NaN
        H, W = selected_band_left.shape[:2]
        cv_full = np.full((H, W, D), np.nan, dtype=np.float32)

        # Expected shrink from configured window size
        ws = int(self.cfg.get("window_size", self._WINDOW_SIZE))
        L = max(0, (ws - 1) // 2)

        # Run backend (returns (Hc, Wc, D) with Hc=H-2L, Wc=W-2L for L conv layers)
        computed_cv = run_mc_cnn_fast(
            selected_band_left,
            selected_band_right,
            disp_min,
            disp_max,
            self._model_path,
            framework=self._framework,
            variant=self._variant,
            provider=self._provider,
            model_name=self.cfg.get("model_name"),
            window_size=ws,  # pass configured window size to backend
        )

        # Validate backend output size vs configured window size
        Hc, Wc = computed_cv.shape[:2]
        expected_Hc, expected_Wc = H - 2 * L, W - 2 * L
        if (Hc, Wc) != (expected_Hc, expected_Wc):
            raise ValueError(
                f"MC-CNN backend output shape mismatch: got ({Hc},{Wc}), expected ({expected_Hc},{expected_Wc}) "
                f"for window_size={ws} (L={(ws - 1)//2}). "
                f"Check that your weights/model_path correspond to the configured window size."
            )

        # Place backend CV centered in the full canvas using the true net offset (which equals L if consistent)
        off_h = max(0, (H - Hc) // 2)
        off_w = max(0, (W - Wc) // 2)
        cv_full[off_h:off_h + Hc, off_w:off_w + Wc, :] = computed_cv

        # Select requested columns
        index_col = np.asarray(cost_volume.attrs["col_to_compute"])
        # Rebase if first column coordinate != 0
        index_col = index_col - img_left.coords["col"].data[0]

        # Fill Pandora cost volume slice (row, col_to_compute, disp)
        cost_volume["cost_volume"].data = cv_full[:, index_col, :]

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

    :param image_dataset: xr.Dataset with band_im coordinate
    :param band_name: band name to extract. If None, return all bands values.
    :return: selected band data as numpy array (H, W)
    """
    selection = image_dataset if band_name is None else image_dataset.sel(band_im=band_name)
    return selection["im"].to_numpy()
