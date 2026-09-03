#!/usr/bin/env python3
"""Run SteepGS independently on OmniScene bins with durable scene recovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comp_svfgs.dataset_omniscene import (  # noqa: E402
    CENTER150_SAMPLE_COUNT,
    DATA_VERSION,
    OmniSceneDataset,
    resolve_data_root,
    sha256_file,
)
from comp_svfgs.omniscene_preprocess import (  # noqa: E402
    PREPARED_FORMAT_VERSION,
    preprocess_scene,
    validate_prepared_scene,
)
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402


DEFAULT_ITERATIONS = 10_000
DEFAULT_EVAL_ITERATIONS = (1_000, 5_000, 10_000)
DEFAULT_RESOLUTION = (112, 200)
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
PROTOCOL_VERSION = 1
COMPLETION_VERSION = 1
METRIC_NAMES = ("psnr", "ssim", "lpips")
RESERVED_EXTRA_ARGS = {
    "-s", "--source_path", "-m", "--model_path", "-r", "--resolution",
    "--eval", "--no_gui", "--iterations", "--test_iterations", "--save_iterations",
    "--checkpoint_iterations", "--start_checkpoint", "--densify_strategy",
    "--S_estimator", "--omniscene_protocol", "--training_seed", "--gpu",
}

# Only experiment semantics decide whether an existing result may be resumed.
# Git/code/software identities remain in protocol.json for provenance, but a
# commit or an environment metadata change must not invalidate completed scenes.
RESULT_RESUME_FIELDS = (
    "protocol_version",
    "mode",
    "data_version",
    "split_sha256",
    "sample_count",
    "resolution",
    "confidence_threshold",
    "train_view_count",
    "target_view_count",
    "iterations",
    "eval_iterations",
    "seed",
    "state_semantics",
    "metric_protocol",
    "training_command_contract",
    "effective_training_parameters",
    "determinism",
)
SCENE_RESUME_FIELDS = (
    "scene_name",
    "bin_token",
    "split_index",
    "prepared_manifest_fingerprint",
    "context_view_ids",
    "target_view_ids",
    "source_numeric_fingerprint",
)


def parse_resolution(value: str) -> Tuple[int, int]:
    try:
        height, width = value.lower().split("x", 1)
        resolution = int(height), int(width)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Resolution must be HxW, e.g. 112x200") from exc
    if resolution not in ((112, 200), (224, 400)):
        raise argparse.ArgumentTypeError("Resolution must be 112x200 or 224x400")
    return resolution


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _git_output(arguments: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=str(REPO_ROOT), check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _code_identities() -> Dict[str, str]:
    relative_paths = (
        "comp_svfgs/dataset_omniscene.py",
        "comp_svfgs/omniscene_preprocess.py",
        "comp_svfgs/omniscene_evaluation.py",
        "scene/dataset_readers.py",
        "scene/__init__.py",
        "train.py",
        "arguments/__init__.py",
        "scripts/run_omniscene.py",
    )
    return {relative: sha256_file(REPO_ROOT / relative) for relative in relative_paths}


def _effective_training_parameters(
    iterations: int, seed: int, extra_train_args: Sequence[str]
) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    model_group = ModelParams(parser)
    optimization_group = OptimizationParams(parser)
    pipeline_group = PipelineParams(parser)
    known, unknown = parser.parse_known_args(list(extra_train_args))

    def values(group: Any) -> Dict[str, Any]:
        return {
            key.lstrip("_"): getattr(known, key.lstrip("_"))
            for key in vars(group)
        }

    model = values(model_group)
    optimization = values(optimization_group)
    pipeline = values(pipeline_group)
    model.update({"resolution": 1, "eval": True})
    optimization.update(
        {"iterations": int(iterations), "densify_strategy": ["steepest"], "S_estimator": "inv_cov"}
    )
    return {
        "model": model,
        "optimization": optimization,
        "pipeline": pipeline,
        "training": {
            "no_gui": True,
            "checkpoint_iterations": [],
            "start_checkpoint": None,
            "training_seed": int(seed),
        },
        "unparsed_extra_args": unknown,
    }


def build_protocol(
    dataset: OmniSceneDataset,
    resolution: Tuple[int, int],
    confidence_threshold: float,
    iterations: int,
    eval_iterations: Sequence[int],
    seed: int,
    extra_train_args: Sequence[str],
) -> Dict[str, Any]:
    git_status = _git_output(["status", "--porcelain=v1", "--untracked-files=all"])
    payload: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": dataset.mode,
        "data_version": DATA_VERSION,
        "data_root": str(dataset.data_root),
        "split_path": str(dataset.split_path) if dataset.split_path else None,
        "split_sha256": dataset.split_sha256,
        "sample_count": len(dataset),
        "resolution": list(resolution),
        "confidence_threshold": float(confidence_threshold),
        "train_view_count": 6,
        "target_view_count": 18,
        "iterations": int(iterations),
        "eval_iterations": list(eval_iterations),
        "seed": int(seed),
        "state_semantics": "iteration k is evaluated before optimizer/densification for k",
        "metric_protocol": {
            "primary": "all_18_macro_by_scene",
            "diagnostic": "novel_12_macro_by_scene",
            "metrics": list(METRIC_NAMES),
            "mask": None,
        },
        "training_command_contract": {
            "resolution_argument": 1,
            "densify_strategy": "steepest",
            "S_estimator": "inv_cov",
            "checkpointing": False,
            "extra_train_args": list(extra_train_args),
        },
        "effective_training_parameters": _effective_training_parameters(
            iterations, seed, extra_train_args
        ),
        "determinism": {
            "python_numpy_torch_seed": int(seed),
            "cudnn_benchmark": False,
            "evaluation_rng_state_restored": True,
        },
        "code_sha256": _code_identities(),
        "git": {
            "commit": _git_output(["rev-parse", "HEAD"]),
            "dirty": bool(git_status),
            "status": git_status,
            "submodules": _git_output(["submodule", "status", "--recursive"]),
        },
        "software": {
            "python": sys.version,
            "torch": _version("torch"),
            "pytorch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "torchvision": _version("torchvision"),
            "numpy": _version("numpy"),
            "pillow": _version("Pillow"),
            "plyfile": _version("plyfile"),
            "lpips_implementation": "bundled lpipsPyTorch LPIPS(vgg, version=0.1)",
        },
    }
    payload["fingerprint"] = _canonical_sha256(payload)
    return payload


def _resume_compatible(
    existing: Dict[str, Any], current: Dict[str, Any], fields: Sequence[str]
) -> bool:
    if existing == current:
        return True
    try:
        return all(existing[field] == current[field] for field in fields)
    except (KeyError, TypeError):
        return False


def _ensure_result_protocol(path: Path, protocol: Dict[str, Any]) -> None:
    if path.is_file():
        with path.open("r", encoding="utf-8") as input_file:
            existing = json.load(input_file)
        if not _resume_compatible(existing, protocol, RESULT_RESUME_FIELDS):
            raise RuntimeError(
                f"Result root contains a different protocol: {path}. "
                "Use a new --result-root or restore the original experiment parameters."
            )
        if existing != protocol:
            print(
                "[RESUME] Git/code/software provenance changed; "
                "reuse the semantically compatible result root",
                flush=True,
            )
        return
    _atomic_write_json(path, protocol)


def _write_preprocess_protocol(path: Path, protocol: Dict[str, Any]) -> None:
    payload = {
        "prepared_format_version": PREPARED_FORMAT_VERSION,
        "mode": protocol["mode"],
        "data_version": protocol["data_version"],
        "data_root": protocol["data_root"],
        "split_path": protocol["split_path"],
        "split_sha256": protocol["split_sha256"],
        "resolution": protocol["resolution"],
        "confidence_threshold": protocol["confidence_threshold"],
        "code_sha256": {
            key: value for key, value in protocol["code_sha256"].items()
            if key.startswith("comp_svfgs/dataset_") or key.startswith("comp_svfgs/omniscene_preprocess")
        },
    }
    payload["fingerprint"] = _canonical_sha256(payload)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing == payload:
            return
    _atomic_write_json(path, payload)


def validate_extra_train_args(parser: argparse.ArgumentParser, values: Sequence[str]) -> None:
    for value in values:
        option = value.split("=", 1)[0]
        if option in RESERVED_EXTRA_ARGS:
            parser.error(f"{option} is protocol-managed and cannot appear in --extra-train-args")


def build_train_command(
    scene_dir: Path,
    model_path: Path,
    iterations: int,
    eval_iterations: Sequence[int],
    seed: int,
    extra_train_args: Sequence[str],
) -> List[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--eval",
        "-s", str(scene_dir),
        "-m", str(model_path),
        "-r", "1",
        "--no_gui",
        "--iterations", str(iterations),
        "--test_iterations", *[str(value) for value in eval_iterations],
        "--save_iterations", str(iterations),
        "--densify_strategy", "steepest",
        "--S_estimator", "inv_cov",
        "--omniscene_protocol",
        "--training_seed", str(seed),
        *extra_train_args,
    ]
    joined = " ".join(command)
    if "steepest" not in command or "inv_cov" not in command or "--omniscene_protocol" not in command:
        raise RuntimeError("Constructed command is missing required SteepGS protocol options")
    if "checkpoint" in joined or "start_checkpoint" in joined:
        raise RuntimeError("Constructed command unexpectedly enables checkpoints")
    return command


def _parse_metrics_text(path: Path) -> Dict[str, float]:
    expected = {"PSNR": "psnr", "SSIM": "ssim", "LPIPS": "lpips"}
    values: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            name, separator, value = line.partition(":")
            if separator and name.strip() in expected:
                values[expected[name.strip()]] = float(value.strip())
    if set(values) != set(METRIC_NAMES) or not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Invalid metrics TXT: {path}")
    return values


def _parse_training_time(path: Path) -> float:
    name, separator, value = path.read_text(encoding="utf-8").strip().partition(":")
    if not separator or name.strip() != "TRAINING_TIME_SECONDS":
        raise ValueError(f"Invalid training-time file: {path}")
    seconds = float(value.strip())
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"Invalid training time: {path}")
    return seconds


def _load_expected_test_views(scene_dir: Path) -> Tuple[List[str], Tuple[int, int]]:
    transforms = json.loads((scene_dir / "transforms_test.json").read_text(encoding="utf-8"))
    frames = transforms.get("frames", [])
    view_ids = [frame.get("view_id") for frame in frames]
    if len(view_ids) != 18 or len(set(view_ids)) != 18 or not all(
        isinstance(view_id, str) for view_id in view_ids
    ):
        raise ValueError("Prepared test transforms must contain 18 unique view IDs")
    dimensions = {(frame.get("height"), frame.get("width")) for frame in frames}
    if len(dimensions) != 1:
        raise ValueError("Prepared test views do not share one resolution")
    return view_ids, next(iter(dimensions))


def _validate_image(path: Path, resolution: Tuple[int, int]) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty image: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (resolution[1], resolution[0]):
            raise ValueError(f"Unexpected image mode/size: {path}")


def _validate_final_ply(path: Path) -> int:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing final Gaussian PLY: {path}")
    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    if len(vertices) <= 0:
        raise ValueError(f"Empty final Gaussian PLY: {path}")
    property_names = [prop.name for prop in vertices.properties]
    for required in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"):
        if required not in property_names:
            raise ValueError(f"Final PLY lacks {required}: {path}")
    for name in property_names:
        values = np.asarray(vertices[name])
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            raise ValueError(f"Final PLY contains non-finite {name}: {path}")
    return len(vertices)


def _read_iteration_result(
    model_path: Path, scene_dir: Path, iteration: int
) -> Dict[str, Any]:
    view_ids, resolution = _load_expected_test_views(scene_dir)
    metrics_path = model_path / f"metrics_{iteration}.json"
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    if metrics.get("iteration") != iteration or metrics.get("target_view_count") != 18:
        raise ValueError(f"Metrics JSON metadata mismatch: {metrics_path}")
    records = metrics.get("per_view")
    if not isinstance(records, list) or [record.get("view_id") for record in records] != view_ids:
        raise ValueError(f"Metrics view order mismatch: {metrics_path}")
    for record in records:
        if not all(math.isfinite(float(record.get(name))) for name in METRIC_NAMES):
            raise ValueError(f"Non-finite per-view metric: {metrics_path}")
    expected_all = {
        name: mean(float(record[name]) for record in records) for name in METRIC_NAMES
    }
    expected_novel = {
        name: mean(float(record[name]) for record in records[:12]) for name in METRIC_NAMES
    }
    for group, expected in (("all_18", expected_all), ("novel_12", expected_novel)):
        values = metrics.get(group, {})
        for name in METRIC_NAMES:
            if not math.isclose(float(values.get(name)), expected[name], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Metric average mismatch for {group}/{name}: {metrics_path}")
    text_values = _parse_metrics_text(model_path / f"metrics_{iteration}.txt")
    for name in METRIC_NAMES:
        if not math.isclose(text_values[name], expected_all[name], rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"Metrics JSON/TXT mismatch for {name}: {metrics_path}")
    training_time = _parse_training_time(model_path / f"training_time_{iteration}.txt")
    if not math.isclose(
        training_time, float(metrics.get("training_time_seconds")), rel_tol=1e-8, abs_tol=1e-8
    ):
        raise ValueError(f"Metrics/timing mismatch: {metrics_path}")

    iteration_dir = model_path / "test" / f"ours_{iteration}"
    expected_names = {view_id + ".png" for view_id in view_ids}
    for directory_name in ("renders", "gt"):
        directory = iteration_dir / directory_name
        actual_names = {path.name for path in directory.glob("*.png")} if directory.is_dir() else set()
        if actual_names != expected_names:
            raise ValueError(f"Expected exactly 18 {directory_name} images at iteration {iteration}")
        for filename in expected_names:
            _validate_image(directory / filename, resolution)
    return metrics


def _result_artifacts(model_path: Path, iterations: int, eval_iterations: Sequence[int]) -> List[Path]:
    paths = [
        model_path / "cfg_args",
        model_path / "train_log.txt",
        model_path / "protocol.json",
        model_path / "omniscene_training_trace.json",
    ]
    for iteration in eval_iterations:
        paths.extend(
            [
                model_path / f"metrics_{iteration}.json",
                model_path / f"metrics_{iteration}.txt",
                model_path / f"training_time_{iteration}.txt",
            ]
        )
        paths.extend(sorted((model_path / "test" / f"ours_{iteration}" / "renders").glob("*.png")))
        paths.extend(sorted((model_path / "test" / f"ours_{iteration}" / "gt").glob("*.png")))
    paths.append(model_path / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply")
    return paths


def _artifact_manifest(
    model_path: Path, iterations: int, eval_iterations: Sequence[int]
) -> List[Dict[str, Any]]:
    identities = []
    for path in _result_artifacts(model_path, iterations, eval_iterations):
        if not path.is_file():
            raise ValueError(f"Missing result artifact: {path}")
        identities.append(
            {
                "path": path.relative_to(model_path).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return identities


def validate_scene_result(
    model_path: Path,
    scene_dir: Path,
    bin_token: str,
    split_index: int,
    resolution: Tuple[int, int],
    confidence_threshold: float,
    split_sha256: str,
    iterations: int,
    eval_iterations: Sequence[int],
    expected_protocol: Dict[str, Any],
    require_completion: bool = True,
) -> Tuple[bool, str]:
    try:
        prepared, reason = validate_prepared_scene(
            scene_dir, bin_token, split_index, resolution, confidence_threshold, split_sha256
        )
        if not prepared:
            return False, f"prepared cache invalid: {reason}"
        protocol = json.loads((model_path / "protocol.json").read_text(encoding="utf-8"))
        if not _resume_compatible(protocol, expected_protocol, SCENE_RESUME_FIELDS):
            return False, "scene experiment semantics mismatch"
        if list(model_path.glob("chkpnt*.pth")):
            return False, "checkpoint files are forbidden by the OmniScene protocol"
        times = []
        for iteration in eval_iterations:
            _read_iteration_result(model_path, scene_dir, iteration)
            times.append(_parse_training_time(model_path / f"training_time_{iteration}.txt"))
        if times != sorted(times):
            return False, "cumulative pure training times are not monotonic"
        trace = json.loads(
            (model_path / "omniscene_training_trace.json").read_text(encoding="utf-8")
        )
        if trace.get("iterations") != iterations or trace.get("selection_count") != iterations:
            return False, "training trace iteration count mismatch"
        view_sequence_sha256 = trace.get("view_sequence_sha256")
        if (
            not isinstance(view_sequence_sha256, str)
            or len(view_sequence_sha256) != 64
            or any(character not in "0123456789abcdef" for character in view_sequence_sha256)
        ):
            return False, "invalid training camera sequence fingerprint"
        milestone_counts = trace.get("gaussian_count_at_milestones")
        if not isinstance(milestone_counts, dict) or set(milestone_counts) != {
            str(iteration) for iteration in eval_iterations
        }:
            return False, "training trace milestone set mismatch"
        if any(not isinstance(value, int) or value <= 0 for value in milestone_counts.values()):
            return False, "invalid Gaussian count in training trace"
        final_point_count = _validate_final_ply(
            model_path / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        )
        if final_point_count != milestone_counts[str(iterations)]:
            return False, "final PLY count differs from the final evaluated Gaussian count"
        if require_completion:
            completion_path = model_path / "center150_complete.json"
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("completion_version") != COMPLETION_VERSION:
                return False, "completion marker version mismatch"
            if completion.get("protocol_fingerprint") != protocol.get("fingerprint"):
                return False, "completion marker does not match its recorded scene protocol"
            artifact_manifest = _artifact_manifest(model_path, iterations, eval_iterations)
            if completion.get("artifact_fingerprint") != _canonical_sha256(artifact_manifest):
                return False, "completion artifact fingerprint mismatch"
        return True, "complete"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)


def _safe_remove_scene_directory(path: Path, experiment_root: Path, allowed_names: Sequence[str]) -> None:
    path = path.resolve()
    experiment_root = experiment_root.resolve()
    if path.parent != experiment_root or path.name not in set(allowed_names):
        raise ValueError(f"Refusing unsafe scene cleanup: {path}")
    if path.is_dir():
        shutil.rmtree(str(path))


def _run_training(command: Sequence[str], gpu: str, log_path: Path) -> None:
    print("[RUN] " + " ".join(shlex.quote(value) for value in command), flush=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command), cwd=str(REPO_ROOT), env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _scene_protocol(global_protocol: Dict[str, Any], scene_dir: Path, scene_name: str, token: str) -> Dict[str, Any]:
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "global_protocol_fingerprint": global_protocol["fingerprint"],
        "scene_name": scene_name,
        "bin_token": token,
        "split_index": manifest["split_index"],
        "prepared_manifest_fingerprint": manifest["manifest_fingerprint"],
        "context_view_ids": manifest["context_view_ids"],
        "target_view_ids": manifest["target_view_ids"],
        "source_numeric_fingerprint": manifest["numeric_fingerprint"],
    }
    payload["fingerprint"] = _canonical_sha256(payload)
    return payload


def _write_completion(
    model_path: Path,
    scene_protocol: Dict[str, Any],
    iterations: int,
    eval_iterations: Sequence[int],
) -> None:
    artifacts = _artifact_manifest(model_path, iterations, eval_iterations)
    payload = {
        "completion_version": COMPLETION_VERSION,
        "scene_name": scene_protocol["scene_name"],
        "bin_token": scene_protocol["bin_token"],
        "protocol_fingerprint": scene_protocol["fingerprint"],
        "artifact_fingerprint": _canonical_sha256(artifacts),
        "artifacts": artifacts,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(model_path / "center150_complete.json", payload)


def _record_failure(failure_root: Path, scene_name: str, message: str) -> None:
    failure_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        failure_root / f"{scene_name}.json",
        {"scene_name": scene_name, "error": message, "time": datetime.now(timezone.utc).isoformat()},
    )


def _prepare_if_needed(
    dataset: OmniSceneDataset,
    index: int,
    scene_dir: Path,
    prepared_root: Path,
    confidence_threshold: float,
) -> Path:
    token = dataset.bin_tokens[index]
    valid, reason = validate_prepared_scene(
        scene_dir,
        token,
        index,
        dataset.resolution,
        confidence_threshold,
        dataset.split_sha256,
    )
    if valid:
        print(f"[PREPARED] reuse cache: {scene_dir.name}", flush=True)
        return scene_dir
    print(f"[PREPARE] rebuild {scene_dir.name}: {reason}", flush=True)
    return preprocess_scene(dataset[index], prepared_root, confidence_threshold)


def _run_scene(
    dataset: OmniSceneDataset,
    index: int,
    prepared_root: Path,
    experiment_root: Path,
    global_protocol: Dict[str, Any],
    confidence_threshold: float,
    iterations: int,
    eval_iterations: Sequence[int],
    seed: int,
    gpu: str,
    extra_train_args: Sequence[str],
) -> Tuple[str, str, Path, Path]:
    token = dataset.bin_tokens[index]
    scene_name = f"{index + 1:03d}_{token}"
    scene_dir = prepared_root / scene_name
    final_path = experiment_root / scene_name
    work_path = experiment_root / f"{scene_name}.work"
    failure_root = experiment_root / "failures"

    # Completion is checked before dataset[index], so a complete scene never
    # decodes source RGB/depth arrays.
    if scene_dir.is_dir() and final_path.is_dir():
        try:
            expected_protocol = _scene_protocol(global_protocol, scene_dir, scene_name, token)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            expected_protocol = {}
        complete, reason = validate_scene_result(
            final_path, scene_dir, token, index, tuple(global_protocol["resolution"]),
            confidence_threshold, dataset.split_sha256, iterations, eval_iterations,
            expected_protocol, require_completion=True,
        )
        if complete:
            print(f"[SKIP] complete scene: {scene_name}", flush=True)
            return scene_name, token, scene_dir, final_path
        print(f"[RESTART] incomplete scene {scene_name}: {reason}", flush=True)

    scene_dir = _prepare_if_needed(
        dataset, index, scene_dir, prepared_root, confidence_threshold
    )
    scene_protocol = _scene_protocol(global_protocol, scene_dir, scene_name, token)
    complete, reason = validate_scene_result(
        final_path, scene_dir, token, index, tuple(global_protocol["resolution"]),
        confidence_threshold, dataset.split_sha256, iterations, eval_iterations,
        scene_protocol, require_completion=True,
    )
    if complete:
        print(f"[SKIP] complete scene: {scene_name}", flush=True)
        return scene_name, token, scene_dir, final_path

    if dataset.mode == "center150":
        for summary_name in ("center150_metrics_summary.json", "center150_metrics_summary.txt"):
            summary_path = experiment_root / summary_name
            if summary_path.exists():
                summary_path.unlink()
    if final_path.exists():
        _record_failure(failure_root, scene_name, f"discarded incomplete stable result: {reason}")
        _safe_remove_scene_directory(final_path, experiment_root, [scene_name])
    if work_path.exists():
        _record_failure(failure_root, scene_name, "discarded incomplete work directory before restart")
        _safe_remove_scene_directory(work_path, experiment_root, [work_path.name])
    work_path.mkdir()
    _atomic_write_json(work_path / "protocol.json", scene_protocol)
    command = build_train_command(
        scene_dir, work_path, iterations, eval_iterations, seed, extra_train_args
    )
    log_path = failure_root / f"{scene_name}.latest.log"
    failure_root.mkdir(parents=True, exist_ok=True)
    try:
        _run_training(command, gpu, log_path)
    except Exception as exc:
        _record_failure(failure_root, scene_name, repr(exc))
        raise

    complete, reason = validate_scene_result(
        work_path, scene_dir, token, index, tuple(global_protocol["resolution"]),
        confidence_threshold, dataset.split_sha256, iterations, eval_iterations,
        scene_protocol, require_completion=False,
    )
    if not complete:
        _record_failure(failure_root, scene_name, f"post-training validation failed: {reason}")
        raise RuntimeError(f"Incomplete scene artifacts for {scene_name}: {reason}")
    _write_completion(work_path, scene_protocol, iterations, eval_iterations)
    complete, reason = validate_scene_result(
        work_path, scene_dir, token, index, tuple(global_protocol["resolution"]),
        confidence_threshold, dataset.split_sha256, iterations, eval_iterations,
        scene_protocol, require_completion=True,
    )
    if not complete:
        raise RuntimeError(f"Completion validation failed for {scene_name}: {reason}")
    os.replace(str(work_path), str(final_path))
    print(f"[DONE] {scene_name}", flush=True)
    return scene_name, token, scene_dir, final_path


def aggregate_center150_results(
    experiment_root: Path,
    records: Sequence[Tuple[str, str, Path, Path]],
    dataset: OmniSceneDataset,
    global_protocol: Dict[str, Any],
    confidence_threshold: float,
    iterations: int,
    eval_iterations: Sequence[int],
) -> Dict[str, Any]:
    if len(records) != CENTER150_SAMPLE_COUNT:
        raise RuntimeError(f"Center150 summary requires {CENTER150_SAMPLE_COUNT} scene records")
    samples: List[Dict[str, Any]] = []
    accumulators = {
        iteration: {
            "all_18": {name: [] for name in METRIC_NAMES},
            "novel_12": {name: [] for name in METRIC_NAMES},
            "training_time_seconds": [],
        }
        for iteration in eval_iterations
    }
    for index, (scene_name, token, scene_dir, model_path) in enumerate(records):
        scene_protocol = _scene_protocol(global_protocol, scene_dir, scene_name, token)
        complete, reason = validate_scene_result(
            model_path, scene_dir, token, index, tuple(global_protocol["resolution"]),
            confidence_threshold, dataset.split_sha256, iterations, eval_iterations,
            scene_protocol, require_completion=True,
        )
        if not complete:
            raise RuntimeError(f"Cannot aggregate incomplete scene {scene_name}: {reason}")
        sample_iterations: Dict[str, Any] = {}
        for iteration in eval_iterations:
            metrics = _read_iteration_result(model_path, scene_dir, iteration)
            sample_iterations[str(iteration)] = metrics
            for group in ("all_18", "novel_12"):
                for name in METRIC_NAMES:
                    accumulators[iteration][group][name].append(metrics[group][name])
            accumulators[iteration]["training_time_seconds"].append(
                metrics["training_time_seconds"]
            )
        samples.append({"scene_name": scene_name, "bin_token": token, "iterations": sample_iterations})

    milestones: Dict[str, Any] = {}
    for iteration in eval_iterations:
        milestone: Dict[str, Any] = {}
        for group in ("all_18", "novel_12"):
            milestone[group] = {
                name: {
                    "mean": mean(accumulators[iteration][group][name]),
                    "std": pstdev(accumulators[iteration][group][name]),
                }
                for name in METRIC_NAMES
            }
        times = accumulators[iteration]["training_time_seconds"]
        milestone["training_time_seconds"] = {"mean": mean(times), "std": pstdev(times)}
        milestones[str(iteration)] = milestone
    summary = {
        "schema_version": 1,
        "protocol_fingerprint": global_protocol["fingerprint"],
        "split": "center150",
        "sample_count": CENTER150_SAMPLE_COUNT,
        "milestones": milestones,
        "samples": samples,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(experiment_root / "center150_metrics_summary.json", summary)
    lines = [f"Center150 scenes: {CENTER150_SAMPLE_COUNT}"]
    for iteration in eval_iterations:
        result = milestones[str(iteration)]
        lines.append(f"Iteration {iteration}")
        for name in METRIC_NAMES:
            item = result["all_18"][name]
            lines.append(f"{name.upper()}: {item['mean']:.7f} (std {item['std']:.7f})")
        timing = result["training_time_seconds"]
        lines.append(
            f"TRAINING_TIME_SECONDS: {timing['mean']:.7f} (std {timing['std']:.7f})"
        )
    _atomic_write_text(
        experiment_root / "center150_metrics_summary.txt", "\n".join(lines) + "\n"
    )
    return summary


def _experiment_tag(
    mode: str,
    resolution: Tuple[int, int],
    confidence_threshold: float,
    iterations: int,
    eval_iterations: Sequence[int],
    seed: int,
    extra_args: Sequence[str],
) -> str:
    tag = f"{mode}_{resolution[0]}x{resolution[1]}"
    if confidence_threshold != DEFAULT_CONFIDENCE_THRESHOLD:
        tag += f"_conf{confidence_threshold:g}"
    non_default = (
        iterations != DEFAULT_ITERATIONS
        or tuple(eval_iterations) != DEFAULT_EVAL_ITERATIONS
        or seed != 0
        or bool(extra_args)
    )
    if non_default:
        payload = {
            "iterations": iterations,
            "eval": list(eval_iterations),
            "seed": seed,
            "extra": list(extra_args),
        }
        tag += "_" + _canonical_sha256(payload)[:10]
    return tag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SteepGS OmniScene per-bin optimization, evaluation, recovery, and aggregation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--mode", choices=["train", "val", "test", "demo", "center150"], default="center150")
    parser.add_argument("--resolution", type=parse_resolution, default=DEFAULT_RESOLUTION)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--eval-iterations", type=int, nargs="+", default=list(DEFAULT_EVAL_ITERATIONS))
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--preprocessed-root", type=Path, default=REPO_ROOT / "output" / "omniscene_preprocessed")
    parser.add_argument("--result-root", type=Path, default=REPO_ROOT / "output" / "omniscene_results")
    parser.add_argument(
        "--scene-indices", nargs="+", type=int, default=None,
        help="Optional 1-based subset for smoke/debug; formal Center150 omits this option",
    )
    parser.add_argument("--extra-train-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    eval_iterations = tuple(args.eval_iterations)
    if (
        not eval_iterations
        or tuple(sorted(set(eval_iterations))) != eval_iterations
        or eval_iterations[0] <= 0
        or eval_iterations[-1] != args.iterations
    ):
        parser.error("--eval-iterations must be unique/increasing/positive and end at --iterations")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be in [0,1]")
    validate_extra_train_args(parser, args.extra_train_args)

    data_root = resolve_data_root(args.data_root)
    dataset = OmniSceneDataset(data_root, args.mode, args.resolution)
    tag = _experiment_tag(
        args.mode, args.resolution, args.confidence_threshold, args.iterations,
        eval_iterations, args.seed, args.extra_train_args,
    )
    prepared_tag = f"{args.mode}_{args.resolution[0]}x{args.resolution[1]}"
    if args.confidence_threshold != DEFAULT_CONFIDENCE_THRESHOLD:
        prepared_tag += f"_conf{args.confidence_threshold:g}"
    prepared_root = (args.preprocessed_root / prepared_tag).resolve()
    experiment_root = (args.result_root / tag).resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    protocol = build_protocol(
        dataset, args.resolution, args.confidence_threshold, args.iterations,
        eval_iterations, args.seed, args.extra_train_args,
    )
    _write_preprocess_protocol(prepared_root / "protocol.json", protocol)
    _ensure_result_protocol(experiment_root / "protocol.json", protocol)

    all_indices = list(range(len(dataset)))
    if args.scene_indices is not None:
        if any(value < 1 or value > len(dataset) for value in args.scene_indices):
            parser.error(f"--scene-indices must be within 1..{len(dataset)}")
        if len(set(args.scene_indices)) != len(args.scene_indices):
            parser.error("--scene-indices cannot contain duplicates")
        selected_indices = [value - 1 for value in args.scene_indices]
    else:
        selected_indices = all_indices

    records_by_index: Dict[int, Tuple[str, str, Path, Path]] = {}
    for index in selected_indices:
        record = _run_scene(
            dataset, index, prepared_root, experiment_root, protocol,
            args.confidence_threshold, args.iterations, eval_iterations,
            args.seed, args.gpu, args.extra_train_args,
        )
        records_by_index[index] = record

    if args.mode == "center150" and selected_indices == all_indices:
        records = [records_by_index[index] for index in all_indices]
        aggregate_center150_results(
            experiment_root, records, dataset, protocol, args.confidence_threshold,
            args.iterations, eval_iterations,
        )
        print(f"[SUMMARY] wrote {experiment_root / 'center150_metrics_summary.json'}")
    elif args.mode == "center150":
        print("[INFO] subset run complete; global Center150 summary intentionally not written")


if __name__ == "__main__":
    main()
