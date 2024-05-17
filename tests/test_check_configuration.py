# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions to test the plugin configuration.
"""

import json_checker
import pytest

from pandora import matching_cost
from mc_cnn.weights import get_weights


@pytest.fixture()
def basic_cfg():
    return {"matching_cost_method": "mc_cnn"}


@pytest.fixture()
def model_path():
    return str(get_weights())


class TestWindowSize:
    """
    Test window_size in mc_cnn plugin configuration
    """

    @pytest.mark.parametrize("window_cfg", [{}, {"window_size": 11}])
    def test_nominal_case(self, window_cfg, basic_cfg):
        basic_cfg.update(window_cfg)
        matching_cost.AbstractMatchingCost(**basic_cfg)

    @pytest.mark.parametrize(
        "window_cfg",
        [
            pytest.param({"window_size": "eleven"}, id="string value"),
            pytest.param({"window_size": 1}, id="positive value different from eleven"),
            pytest.param({"window_size": -1}, id="negative value different from eleven"),
            pytest.param({"window_size": [1]}, id="list value"),
            pytest.param({"window_size": {"value": 1}}, id="dict value"),
        ],
    )
    def test_fails(self, window_cfg, basic_cfg):
        basic_cfg.update(window_cfg)
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.AbstractMatchingCost(**basic_cfg)
        assert 'key="window_size"' in err.value.args[0]


class TestModelPath:
    """
    Test model_path in mc_cnn plugin configuration
    """

    def test_nominal_case_without_param(self, basic_cfg):
        matching_cost.AbstractMatchingCost(**basic_cfg)

    def test_nominal_case_with_param(self, model_path, basic_cfg):
        basic_cfg.update({"model_path": model_path})
        matching_cost.AbstractMatchingCost(**basic_cfg)

    @pytest.mark.parametrize(
        "model_path_cfg",
        [
            pytest.param({"model_path": "eleven"}, id="path does not exist"),
            pytest.param({"model_path": 1}, id="integer value"),
            pytest.param({"model_path": 1.0}, id="float value"),
            pytest.param({"model_path": ["mc_cnn_fast_mb_weights.pt"]}, id="list value"),
            pytest.param({"model_path": {"value": "mc_cnn_fast_mb_weights.pt"}}, id="dict value"),
        ],
    )
    def test_fails(self, model_path_cfg, basic_cfg):
        basic_cfg.update(model_path_cfg)
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.AbstractMatchingCost(**basic_cfg)
        assert 'key="model_path"' in err.value.args[0]
