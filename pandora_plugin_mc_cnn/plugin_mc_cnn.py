#!/usr/bin/env python
# coding: utf8
"""
Pandora matching_cost plugin for MC-CNN (CPU-only, frameworks + variants).
Supports window_size 7/11/13/15.

Emits (stdout for fallback parsing):
  - PROFILING_TOTAL_CV: time=...s, mem_peak=...MB, framework=..., variant=..., provider=..., window=...

Also writes structured total-stage metrics to:
  <PANDORA_RUN_OUTPUT_DIR>/metrics_total.json
with:
  - total_cv_time, total_cv_mem, framework, provider, variant, window_size, model_path

Framework (CPU):
  - pytorch
  - onnx
  - openvino

Variant:
  - baseline
  - opt1, opt1_notorch
  - opt2, opt2_notorch
  - cpp, cpp2, cpp_notorch, cpp2_notorch

Provider:
  - cpu_base
  - openvino
"""

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
    """Convert bytes to megabytes."""
    return b / (1024.0 * 1024.0)


class MemorySampler:
    """
    Background thread to capture peak RSS during MC-CNN execution.
    
    Configurable via MCCNN_MEM_SAMPLE_SEC (default: 0.005s).
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
        """Background sampling loop."""
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
        """Start memory sampling."""
        self._peak = get_memory_usage_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mem_sampler_total", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop memory sampling."""
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join()
            except Exception:
                pass

    @property
    def peak_mb(self) -> float:
        """Get peak memory in MB."""
        return bytes_to_mb(self._peak)


def _write_metrics_total(
    framework: str,
    provider: str,
    variant: str,
    window_size: Optional[int],
    model_path: Optional[str],
    total_time: float,
    total_mem_mb: float
) -> None:
    """Write total-stage metrics to metrics_total.json."""
    out_dir = os.getenv("PANDORA_RUN_OUTPUT_DIR", "")
    if not out_dir:
        return
        
    try:
        p = Path(out_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "framework": framework,
            "provider": provider,
            "variant": variant,
            "window_size": int(window_size) if window_size is not None else None,
            "model_path": model_path,
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
    MC-CNN fast plugin for Pandora.
    
    Computes cost volume using MC-CNN features and similarity matching.
    CPU-only implementation.
    """

    _WINDOW_SIZE = 11
    _SUBPIX = 1
    _MODEL_PATH = str(get_weights())  # Default weights from package
    _BAND = None
    _FRAMEWORK = "pytorch"
    _VARIANT = "baseline"
    _PROVIDER = "cpu_base"

    def __init__(self, **cfg: Union[int, str]):
        """
        Initialize plugin with configuration.
        
        Args:
            window_size: Patch window size (7, 11, 13, or 15)
            model_path: Path to trained weights
            framework: "pytorch", "onnx", or "openvino"
            variant: Implementation variant
            provider: ONNX provider ("cpu_base" or "openvino")
            model_name: Specific model file name
        """
        super().instantiate_class(**cfg)
        self._model_path = str(self.cfg["model_path"])
        self._framework = str(self.cfg.get("framework", self._FRAMEWORK))
        self._variant = str(self.cfg.get("variant", self._VARIANT))
        self._provider = str(self.cfg.get("provider", self._PROVIDER))
        self._window_size = int(self.cfg.get("window_size", self._WINDOW_SIZE))

    def check_conf(self, **cfg: Union[int, str]) -> Dict[str, Union[int, str]]:
        """Validate and fill default configuration values."""
        cfg = super().check_conf(**cfg)

        # Set defaults
        cfg.setdefault("model_path", self._MODEL_PATH)
        cfg.setdefault("framework", self._FRAMEWORK)
        cfg.setdefault("variant", self._VARIANT)
        cfg.setdefault("provider", self._PROVIDER)
        cfg.setdefault("window_size", self._WINDOW_SIZE)

        schema = self.schema
        schema["matching_cost_method"] = And(str, lambda x: x == "mc_cnn")
        schema["window_size"] = And(int, lambda x: x in (7, 11, 13, 15))
        schema["model_path"] = And(str, lambda x: os.path.exists(x))
        schema["framework"] = And(str, lambda x: x in ["pytorch", "onnx", "openvino"])
        schema["variant"] = And(str, lambda x: x in [
            "baseline",
            "opt1", "opt1_notorch",
            "opt2", "opt2_notorch",
            "cpp", "cpp2", "cpp_notorch", "cpp2_notorch",
        ])
        schema["provider"] = And(str, lambda x: x in ["cpu_base", "openvino"])
        
        if "model_name" in cfg:
            schema["model_name"] = str

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
        Compute cost volume for stereo pair.
        
        Emits PROFILING_TOTAL_CV marker and writes metrics_total.json.
        """
        # Start total profiling
        ms_total = MemorySampler().start()
        start_total = time.perf_counter()

        # Validate band input
        self.check_band_input_mc(img_left, img_right)

        # Select band
        selected_band_left = get_band_values(img_left, self._BAND)
        selected_band_right = get_band_values(img_right, self._BAND)

        # Disparity range
        disparity_range = cost_volume.coords["disp"].data
        disp_min, disp_max = int(disparity_range[0]), int(disparity_range[-1])
        offset_row_col = int(cost_volume.attrs["offset_row_col"])

        # Prepare cost volume
        cv = np.full(
            (selected_band_left.shape[0], selected_band_left.shape[1], len(disparity_range)),
            np.nan,
            dtype=np.float32,
        )

        # Compute cost volume (inner region or full frame)
        if offset_row_col != 0:
            cv[offset_row_col:-offset_row_col, offset_row_col:-offset_row_col, :] = run_mc_cnn_fast(
                selected_band_left,
                selected_band_right,
                disp_min,
                disp_max,
                self._model_path,
                framework=self._framework,
                variant=self._variant,
                provider=self._provider,
                model_name=self.cfg.get("model_name"),
            )
        else:
            cv = run_mc_cnn_fast(
                selected_band_left,
                selected_band_right,
                disp_min,
                disp_max,
                self._model_path,
                framework=self._framework,
                variant=self._variant,
                provider=self._provider,
                model_name=self.cfg.get("model_name"),
            )

        # Apply column selection
        index_col = np.asarray(cost_volume.attrs["col_to_compute"])
        index_col = index_col - img_left.coords["col"].data[0]
        cost_volume["cost_volume"].data = cv[:, index_col, :]

        # Update metadata
        cost_volume.attrs.update({
            "type_measure": "min",
            "cmax": 1,
        })

        # Finish profiling
        time_total = time.perf_counter() - start_total
        ms_total.stop()
        mem_total_peak = ms_total.peak_mb

        # Emit marker
        print(
            f"PROFILING_TOTAL_CV: time={time_total:.4f}s, mem_peak={mem_total_peak:.2f}MB, "
            f"framework={self._framework}, variant={self._variant}, provider={self._provider}, window={self._window_size}"
        )

        # Write metrics
        _write_metrics_total(
            self._framework,
            self._provider,
            self._variant,
            self._window_size,
            self._model_path,
            time_total,
            mem_total_peak
        )

        # Optional CV dump for debugging
        if os.getenv("MCCNN_DUMP_CV", "0") in ("1", "true", "True"):
            out_dir_env = os.getenv("PANDORA_RUN_OUTPUT_DIR", "")
            if out_dir_env:
                out_dir = Path(out_dir_env)
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / "mc_cnn_cv.npy", cv)
                except Exception:
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
