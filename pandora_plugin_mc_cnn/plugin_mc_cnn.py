#!/usr/bin/env python
# coding: utf8
#
# Pandora matching_cost plugin for MC-CNN (CPU-only, frameworks + variants).
# Emits (stdout for fallback parsing):
#   - PROFILING_TOTAL_CV: time=...s, mem_peak=...MB, framework=..., variant=...
#
# Also writes structured total-stage metrics to:
#   <PANDORA_RUN_OUTPUT_DIR>/metrics_total.json
# with:
#   - total_cv_time, total_cv_mem
#   - framework, variant
#
# Frameworks (CPU):
#   - pytorch
#   - onnx
#   - openvino
#
# Variants:
#   - baseline
#   - opt1
#
# Providers:
#   - cpu_base
#   - openvino
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
import psutil

from pandora.img_tools import shift_right_img
from pandora.matching_cost import matching_cost
from mc_cnn.run import run_mc_cnn_fast
from mc_cnn.weights import get_weights


def get_memory_usage_bytes() -> int:
    """Get current process RSS in bytes."""
    return psutil.Process().memory_info().rss


def bytes_to_mb(b: int) -> float:
    return b / (1024.0 * 1024.0)


class MemorySampler:
    """
    Background sampler to capture true peak RSS during the total MC-CNN stage.
    Sampling interval can be tuned with env MCCNN_MEM_SAMPLE_SEC (default 0.005s).
    """
    def __init__(self, interval_sec: float = None):
        if interval_sec is None:
            try:
                interval_sec = float(os.getenv("MCCNN_MEM_SAMPLE_SEC", "0.005"))
            except Exception:
                interval_sec = 0.005
        self.interval = max(0.0005, interval_sec)
        self._stop = threading.Event()
        self._thread = None
        self._peak = 0

    def _run(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._peak = get_memory_usage_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mem_sampler_total", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join()
            except Exception:
                pass

    @property
    def peak_mb(self) -> float:
        return bytes_to_mb(self._peak)


def _write_metrics_total(framework: str, variant: str, total_time: float, total_mem_mb: float) -> None:
    """
    Write total-stage metrics JSON to PANDORA_RUN_OUTPUT_DIR if set.
    """
    out_dir = os.getenv("PANDORA_RUN_OUTPUT_DIR", "")
    if not out_dir:
        return
    try:
        p = Path(out_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        payload = {
            "framework": framework,
            "variant": variant,
            "total_cv_time": float(total_time),
            "total_cv_mem": float(total_mem_mb),
        }
        with open(p / "metrics_total.json", "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


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

    def __init__(self, **cfg: Union[int, str]):
        """
        :param cfg: {
            'matching_cost_method': str,
            'window_size': int,
            'subpix': int,
            'model_path': str,
            'framework': 'pytorch'|'onnx'|'openvino',
            'variant': 'baseline'|'opt1'
            'provider': 'cpu_base'|'openvino'
        }
        """
        super().instantiate_class(**cfg)
        self._model_path = str(self.cfg["model_path"])
        self._framework = str(self.cfg.get("frameworks", self._FRAMEWORK))
        self._variant = str(self.cfg.get("variant", self._VARIANT))
        self._provider = str(self.cfg.get("provider", self._PROVIDER))

    def check_conf(self, **cfg: Union[int, str]) -> Dict[str, Union[int, str]]:
        """
        Add defaults and validate configuration.
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
        
        schema = self.schema
        schema["matching_cost_method"] = And(str, lambda x: x == "mc_cnn")
        schema["window_size"] = And(int, lambda x: x == 11)
        schema["model_path"] = And(str, lambda x: os.path.exists(x))
        schema["framework"] = And(str, lambda x: x in ["pytorch", "onnx", "openvino"])
        schema["variant"] = And(str, lambda x: x in ["baseline", "opt1"])
        schema["provider"] = And(str, lambda x: x in ["cpu_base", "openvino"])

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

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
        # Profiling (total) with true peak sampler
        ms_total = MemorySampler().start()
        start_total = time.perf_counter()

        # Check band parameter
        self.check_band_input_mc(img_left, img_right)

        # Select band(s) if multi-band
        selected_band_left = get_band_values(img_left, self._band)
        selected_band_right = get_band_values(img_right, self._band)

        disparity_range = cost_volume.coords["disp"].data
        disp_min, disp_max = int(disparity_range[0]), int(disparity_range[-1])
        offset_row_col = int(cost_volume.attrs["offset_row_col"])

        # Prepare CV container, filled with NaN by default
        cv = np.full(
            (selected_band_left.shape[0], selected_band_left.shape[1], len(disparity_range)),
            np.nan,
            dtype=np.float32,
        )

        # If offset, compute only inner region
        if offset_row_col != 0:
            cv[offset_row_col:-offset_row_col, offset_row_col:-offset_row_col, :] = run_mc_cnn_fast(
                selected_band_left, selected_band_right, disp_min, disp_max, self._model_path,
                framework=self._framework, variant=self._variant
            )
        else:
            cv = run_mc_cnn_fast(
                selected_band_left, selected_band_right, disp_min, disp_max, self._model_path,
                framework=self._framework, variant=self._variant, provider=self._provider
            )

        index_col = np.asarray(cost_volume.attrs["col_to_compute"])
        # Rebase if first column coordinate != 0
        index_col = index_col - img_left.coords["col"].data[0]

        # Fill Pandora cost volume slice (row, col_to_compute, disp)
        cost_volume["cost_volume"].data = cv[:, index_col, :]

        # Metadata
        cost_volume.attrs.update(
            {
                "type_measure": "min",
                "cmax": 1,
            }
        )

        # Stop total timer and sampler BEFORE any optional disk I/O
        time_total = time.perf_counter() - start_total
        ms_total.stop()
        mem_total_peak = ms_total.peak_mb

        # Emit marker
        print(
            f"PROFILING_TOTAL_CV: time={time_total:.4f}s, mem_peak={mem_total_peak:.2f}MB, "
            f"framework={self._framework}, variant={self._variant}"
        )

        # Write structured total metrics
        _write_metrics_total(self._framework, self._variant, time_total, mem_total_peak)

        # Optional dump of the computed CV (for correctness checks) — NOT INCLUDED in timing
        if os.getenv("MCCNN_DUMP_CV", "0") in ("1", "true", "True"):
            out_dir_env = os.getenv("PANDORA_RUN_OUTPUT_DIR", "")
            if out_dir_env:
                out_dir = Path(out_dir_env)
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / "mc_cnn_cv.npy", cv)
                except Exception:
                    # Silent failure on dump to keep baseline simple
                    pass

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
