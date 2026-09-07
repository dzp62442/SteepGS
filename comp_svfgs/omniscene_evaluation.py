"""Side-effect-contained milestone rendering and evaluation for OmniScene."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from lpipsPyTorch.modules.lpips import LPIPS
from comp_svfgs.metric_reporting import METRIC_NAMES, format_metrics_text
from utils.image_utils import psnr
from utils.loss_utils import ssim


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(str(temporary), str(path))


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        output_file.write(content)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(str(temporary), str(path))


def _save_png(image: torch.Tensor, path: Path) -> None:
    array = (
        image.detach().clamp(0.0, 1.0).mul(255.0).round().byte()
        .permute(1, 2, 0).contiguous().cpu().numpy()
    )
    Image.fromarray(array, mode="RGB").save(path, format="PNG")


def _mean(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        raise ValueError("Cannot average an empty view list")
    return {
        metric: float(sum(float(record[metric]) for record in records) / len(records))
        for metric in METRIC_NAMES
    }


class OmniSceneEvaluator:
    """Evaluate the fixed 18 target cameras while preserving every RNG stream."""

    def __init__(self, model_path: str, scene: Any, render_function: Any, render_args: Any):
        self.model_path = Path(model_path).resolve()
        self.scene = scene
        self.render_function = render_function
        self.render_args = render_args
        cameras = scene.getTestCameras()
        if len(cameras) != 18:
            raise ValueError(f"OmniScene evaluation requires 18 target cameras, got {len(cameras)}")
        names = [camera.image_name for camera in cameras]
        if len(set(names)) != 18 or any(
            not name.startswith(f"target_{index:02d}_") for index, name in enumerate(names)
        ):
            raise ValueError("OmniScene target camera identities must be unique and ordered")

        # VGG/LPIPS initialization may consume CPU/CUDA RNG.  It is setup work,
        # not part of training, and must not alter the optimization trajectory.
        rng_state = capture_rng_state()
        try:
            self.lpips_metric = LPIPS(net_type="vgg").to("cuda").eval()
            for parameter in self.lpips_metric.parameters():
                parameter.requires_grad_(False)
        finally:
            restore_rng_state(rng_state)

    def evaluate(self, iteration: int, training_time_seconds: float) -> Dict[str, Any]:
        if not np.isfinite(training_time_seconds) or training_time_seconds < 0.0:
            raise ValueError("Cumulative pure training time must be finite and non-negative")
        rng_state = capture_rng_state()
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".ours_{iteration}.tmp-", dir=str(self.model_path))
        )
        try:
            render_dir = temporary_root / "renders"
            gt_dir = temporary_root / "gt"
            render_dir.mkdir()
            gt_dir.mkdir()
            records: List[Dict[str, Any]] = []
            with torch.no_grad():
                for target_index, viewpoint in enumerate(self.scene.getTestCameras()):
                    rendered = torch.clamp(
                        self.render_function(
                            viewpoint, self.scene.gaussians, *self.render_args
                        )["render"],
                        0.0,
                        1.0,
                    )
                    ground_truth = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    rendered_batch = rendered.unsqueeze(0)
                    ground_truth_batch = ground_truth.unsqueeze(0)
                    record = {
                        "target_index": target_index,
                        "view_id": viewpoint.image_name,
                        "role": "novel" if target_index < 12 else "context",
                        "psnr": float(psnr(rendered_batch, ground_truth_batch).mean().item()),
                        "ssim": float(ssim(rendered_batch, ground_truth_batch).item()),
                        "lpips": float(
                            self.lpips_metric(rendered_batch, ground_truth_batch).mean().item()
                        ),
                    }
                    if not all(np.isfinite(record[name]) for name in METRIC_NAMES):
                        raise ValueError(f"Non-finite metric for {viewpoint.image_name}")
                    records.append(record)
                    filename = viewpoint.image_name + ".png"
                    _save_png(rendered, render_dir / filename)
                    _save_png(ground_truth, gt_dir / filename)

            torch.cuda.synchronize()
            all_metrics = _mean(records)
            novel_metrics = _mean(records[:12])
            payload: Dict[str, Any] = {
                "schema_version": 1,
                "iteration": int(iteration),
                "state_semantics": (
                    "after iteration forward/backward and before that iteration optimizer/densification"
                ),
                "target_view_count": 18,
                "novel_view_count": 12,
                "training_time_seconds": float(training_time_seconds),
                "all_18": all_metrics,
                "novel_12": novel_metrics,
                "per_view": records,
            }

            iteration_root = self.model_path / "test"
            iteration_root.mkdir(exist_ok=True)
            final_directory = iteration_root / f"ours_{iteration}"
            if final_directory.exists():
                shutil.rmtree(str(final_directory))
            os.replace(str(temporary_root), str(final_directory))
            _atomic_write_json(self.model_path / f"metrics_{iteration}.json", payload)
            _atomic_write_text(
                self.model_path / f"metrics_{iteration}.txt",
                format_metrics_text(payload),
            )
            _atomic_write_text(
                self.model_path / f"training_time_{iteration}.txt",
                "TRAINING_TIME_SECONDS: {:.10f}\n".format(training_time_seconds),
            )
            return payload
        finally:
            restore_rng_state(rng_state)
            if temporary_root.exists():
                shutil.rmtree(str(temporary_root))
