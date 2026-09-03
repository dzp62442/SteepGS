"""Prepare one OmniScene bin as a strict, self-describing SteepGS scene."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from comp_svfgs.dataset_omniscene import CAMERA_TYPES, OmniSceneView, sha256_file


PREPARED_FORMAT_VERSION = 1
COORDINATE_CONVENTION = "opencv_camera_to_keyframe_lidar_world"


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


def _file_identity(path: Path, include_sha256: bool = True) -> Dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    identity: Dict[str, Any] = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        identity["sha256"] = sha256_file(path)
    return identity


def _relative_artifact_identity(path: Path, root: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _identity_matches(stored: Dict[str, Any], path: Path) -> bool:
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return False
    if stored.get("path") != str(resolved) or stored.get("size") != int(stat.st_size):
        return False
    if stored.get("mtime_ns") == int(stat.st_mtime_ns):
        return True
    expected_hash = stored.get("sha256")
    return isinstance(expected_hash, str) and sha256_file(resolved) == expected_hash


def _safe_remove_directory(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    parent = parent.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"Refusing to remove path outside expected root: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(str(resolved))


def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor [3,H,W], got {tuple(image.shape)}")
    array = image.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().byte()
    return array.permute(1, 2, 0).contiguous().numpy()


def _save_png_atomic(image: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(_tensor_to_uint8(image), mode="RGB").save(temporary, format="PNG")
    os.replace(str(temporary), str(path))


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError(f"Invalid point cloud shapes: xyz={xyz.shape}, rgb={rgb.shape}")
    if xyz.shape[0] == 0 or not np.isfinite(xyz).all():
        raise ValueError("Initialization point cloud must contain finite points")
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    vertices = np.empty(xyz.shape[0], dtype=dtype)
    normals = np.zeros_like(xyz, dtype=np.float32)
    vertices[:] = list(map(tuple, np.concatenate([xyz, normals, rgb], axis=1)))
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def build_initial_point_cloud(
    context: Dict[str, Any], confidence_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Backproject valid metric z-depth from all six context cameras."""
    images = context["image"]
    depths = context["depth_metric"]
    confidence = context["confidence"]
    intrinsics = context["intrinsics"]
    c2ws = context["c2w"]
    if images.shape[0] != 6:
        raise ValueError(f"Point-cloud initialization requires six context views, got {images.shape[0]}")

    points: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    valid_counts: List[int] = []
    far_counts: List[int] = []
    for image, depth, conf, intrinsic, c2w in zip(images, depths, confidence, intrinsics, c2ws):
        image_np = _tensor_to_uint8(image)
        depth_np = depth.detach().cpu().numpy().astype(np.float32, copy=False)
        conf_np = conf.detach().cpu().numpy().astype(np.float32, copy=False)
        k_np = intrinsic.detach().cpu().numpy().astype(np.float64, copy=False)
        c2w_np = c2w.detach().cpu().numpy().astype(np.float64, copy=False)
        valid = (
            np.isfinite(depth_np)
            & (depth_np > 0.0)
            & np.isfinite(conf_np)
            & (conf_np > float(confidence_threshold))
        )
        valid_counts.append(int(valid.sum()))
        far_counts.append(int((valid & (depth_np > 100.0)).sum()))
        if not valid.any():
            continue
        height, width = depth_np.shape
        v, u = np.indices((height, width), dtype=np.float64)
        z = depth_np.astype(np.float64, copy=False)
        x = (u - k_np[0, 2]) * z / k_np[0, 0]
        y = (v - k_np[1, 2]) * z / k_np[1, 1]
        camera_points = np.stack([x, y, z, np.ones_like(z)], axis=-1)[valid]
        world_points = (c2w_np @ camera_points.T).T[:, :3]
        if not np.isfinite(world_points).all():
            raise ValueError("Backprojection produced non-finite world points")
        points.append(world_points.astype(np.float32))
        colors.append(image_np[valid].astype(np.uint8))

    if not points:
        raise ValueError("No valid depth points; random initialization is intentionally disabled")
    all_points = np.concatenate(points, axis=0)
    all_colors = np.concatenate(colors, axis=0)
    return all_points, all_colors, {
        "valid_points_per_camera": valid_counts,
        "farther_than_100m_per_camera": far_counts,
        "point_count": int(all_points.shape[0]),
    }


def _frame(view: OmniSceneView, file_path: str) -> Dict[str, Any]:
    _, height, width = view.image.shape
    intrinsic = view.intrinsics.detach().cpu().numpy()
    return {
        "view_id": view.view_id,
        "role": view.role,
        "camera_name": view.camera_name,
        "frame_index": int(view.frame_index),
        "file_path": file_path,
        "transform_matrix": view.c2w.detach().cpu().tolist(),
        "fl_x": float(intrinsic[0, 0]),
        "fl_y": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "width": int(width),
        "height": int(height),
    }


def _view_source_record(view: OmniSceneView) -> Dict[str, Any]:
    files = {
        "rgb": _file_identity(view.image_path),
        "intrinsics": _file_identity(view.parameter_path),
    }
    if view.depth_path is not None:
        files["depth_metric"] = _file_identity(view.depth_path)
    if view.confidence_path is not None:
        files["confidence"] = _file_identity(view.confidence_path)
    return {
        "view_id": view.view_id,
        "role": view.role,
        "camera_name": view.camera_name,
        "frame_index": int(view.frame_index),
        "files": files,
        "intrinsics": view.intrinsics.detach().cpu().tolist(),
        "c2w": view.c2w.detach().cpu().tolist(),
    }


def _code_fingerprint() -> Dict[str, str]:
    paths = [Path(__file__).resolve(), Path(__file__).with_name("dataset_omniscene.py").resolve()]
    return {path.name: sha256_file(path) for path in paths}


def _validate_png(path: Path, resolution: Tuple[int, int]) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.mode == "RGB" and image.size == (resolution[1], resolution[0])
    except (OSError, ValueError):
        return False


def validate_prepared_scene(
    scene_dir: Path,
    bin_token: str,
    split_index: int,
    resolution: Tuple[int, int],
    confidence_threshold: float,
    split_sha256: str,
) -> Tuple[bool, str]:
    """Validate cache identity and all prepared scene artifacts."""
    scene_dir = Path(scene_dir)
    manifest_path = scene_dir / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return False, "missing or invalid manifest.json"
    manifest_without_fingerprint = {
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    }
    if manifest.get("manifest_fingerprint") != _canonical_sha256(manifest_without_fingerprint):
        return False, "manifest fingerprint mismatch"

    expected = {
        "format_version": PREPARED_FORMAT_VERSION,
        "bin_token": bin_token,
        "split_index": int(split_index),
        "resolution": list(resolution),
        "confidence_threshold": float(confidence_threshold),
        "split_sha256": split_sha256,
        "coordinate_convention": COORDINATE_CONVENTION,
        "camera_order": list(CAMERA_TYPES),
        "context_view_count": 6,
        "target_view_count": 18,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False, f"manifest mismatch for {key}"
    if manifest.get("code_fingerprint") != _code_fingerprint():
        return False, "preprocessing code fingerprint changed"
    bin_info = manifest.get("bin_info")
    if not isinstance(bin_info, dict) or not isinstance(bin_info.get("path"), str):
        return False, "missing bin-info identity"
    if not _identity_matches(bin_info, Path(bin_info["path"])):
        return False, "bin-info identity changed"

    source_records = manifest.get("source_views")
    if not isinstance(source_records, list) or len(source_records) != 18:
        return False, "manifest source view list is incomplete"
    if [record.get("view_id") for record in source_records] != manifest.get("target_view_ids"):
        return False, "manifest source view order differs from target IDs"
    for source_index, record in enumerate(source_records):
        files = record.get("files", {})
        if not isinstance(files, dict):
            return False, "invalid source file identities"
        expected_file_keys = {"rgb", "intrinsics"}
        if source_index >= 12:
            expected_file_keys.update({"depth_metric", "confidence"})
        if set(files) != expected_file_keys:
            return False, "source file identity set is incomplete"
        for identity in files.values():
            path_value = identity.get("path") if isinstance(identity, dict) else None
            if not isinstance(path_value, str) or not _identity_matches(identity, Path(path_value)):
                return False, f"source identity changed for {path_value}"

    train_path = scene_dir / "transforms_train.json"
    test_path = scene_dir / "transforms_test.json"
    try:
        with train_path.open("r", encoding="utf-8") as train_file:
            train_frames = json.load(train_file)["frames"]
        with test_path.open("r", encoding="utf-8") as test_file:
            test_frames = json.load(test_file)["frames"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False, "invalid transforms files"
    if len(train_frames) != 6 or len(test_frames) != 18:
        return False, "transforms must contain 6 train and 18 test frames"
    train_ids = [frame.get("view_id") for frame in train_frames]
    test_ids = [frame.get("view_id") for frame in test_frames]
    if len(set(train_ids)) != 6 or len(set(test_ids)) != 18:
        return False, "view IDs are not unique"

    referenced_images = {frame.get("file_path") for frame in train_frames + test_frames}
    if None in referenced_images or len(referenced_images) != 18:
        return False, "expected 18 unique PNG files (6 context plus 12 novel)"
    for relative in referenced_images:
        path = scene_dir / relative
        if not _validate_png(path, resolution):
            return False, f"invalid prepared image: {relative}"

    ply_path = scene_dir / "points3D.ply"
    try:
        vertices = PlyData.read(str(ply_path))["vertex"]
        xyz = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
        rgb = np.column_stack([vertices[channel] for channel in ("red", "green", "blue")])
    except (OSError, ValueError, KeyError):
        return False, "invalid points3D.ply"
    if len(vertices) <= 0 or not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        return False, "point cloud is empty or non-finite"
    if manifest.get("point_cloud", {}).get("point_count") != len(vertices):
        return False, "point count differs from manifest"

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return False, "missing artifact identities"
    expected_artifacts = {
        "transforms_train.json", "transforms_test.json", "points3D.ply",
        *referenced_images,
    }
    if {identity.get("path") for identity in artifacts if isinstance(identity, dict)} != expected_artifacts:
        return False, "artifact identity set differs from prepared outputs"
    for identity in artifacts:
        relative = identity.get("path") if isinstance(identity, dict) else None
        if not isinstance(relative, str):
            return False, "invalid artifact identity"
        artifact_path = scene_dir / relative
        try:
            stat = artifact_path.stat()
        except OSError:
            return False, f"missing artifact: {relative}"
        if identity.get("size") != int(stat.st_size):
            return False, f"artifact size changed: {relative}"
        if identity.get("mtime_ns") != int(stat.st_mtime_ns):
            if identity.get("sha256") != sha256_file(artifact_path):
                return False, f"artifact content changed: {relative}"
    return True, "complete"


def prepared_scene_complete(
    scene_dir: Path,
    bin_token: str,
    split_index: int,
    resolution: Tuple[int, int],
    confidence_threshold: float,
    split_sha256: str,
) -> bool:
    return validate_prepared_scene(
        scene_dir, bin_token, split_index, resolution, confidence_threshold, split_sha256
    )[0]


def preprocess_scene(
    scene_data: Dict[str, Any],
    output_root: Path,
    confidence_threshold: float = 0.3,
) -> Path:
    """Build and atomically publish one prepared scene directory."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    split_index = int(scene_data["index"])
    bin_token = scene_data["scene"]
    resolution = tuple(int(value) for value in scene_data["context"]["image"].shape[-2:])
    scene_name = f"{split_index + 1:03d}_{bin_token}"
    scene_dir = output_root / scene_name
    valid, _ = validate_prepared_scene(
        scene_dir,
        bin_token,
        split_index,
        resolution,
        confidence_threshold,
        scene_data["split_sha256"],
    )
    if valid:
        return scene_dir

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{scene_name}.tmp-", dir=str(output_root)))
    try:
        images_dir = temporary_dir / "images"
        images_dir.mkdir()
        context_views: Sequence[OmniSceneView] = scene_data["context"]["views"]
        target_views: Sequence[OmniSceneView] = scene_data["target"]["views"]
        if len(context_views) != 6 or len(target_views) != 18:
            raise ValueError("Prepared OmniScene scenes require exactly 6/18 context/target views")

        context_paths: Dict[str, str] = {}
        for camera_index, view in enumerate(context_views):
            filename = f"context_{camera_index:02d}_{view.camera_name}.png"
            relative = f"images/{filename}"
            _save_png_atomic(view.image, temporary_dir / relative)
            context_paths[view.camera_name] = relative
        novel_paths: Dict[str, str] = {}
        for target_index, view in enumerate(target_views[:12]):
            filename = f"novel_{target_index:02d}_{view.camera_name}_t{view.frame_index}.png"
            relative = f"images/{filename}"
            _save_png_atomic(view.image, temporary_dir / relative)
            novel_paths[view.view_id] = relative

        train_frames = [
            _frame(view, context_paths[view.camera_name]) for view in context_views
        ]
        test_frames = []
        for target_index, view in enumerate(target_views):
            relative = (
                novel_paths[view.view_id]
                if target_index < 12
                else context_paths[view.camera_name]
            )
            test_frames.append(_frame(view, relative))
        transform_header = {
            "format_version": PREPARED_FORMAT_VERSION,
            "coordinate_convention": COORDINATE_CONVENTION,
            "no_flip_yz": True,
        }
        _atomic_write_json(
            temporary_dir / "transforms_train.json",
            dict(transform_header, frames=train_frames),
        )
        _atomic_write_json(
            temporary_dir / "transforms_test.json",
            dict(transform_header, frames=test_frames),
        )

        points, colors, point_summary = build_initial_point_cloud(
            scene_data["context"], confidence_threshold
        )
        _write_ply(temporary_dir / "points3D.ply", points, colors)

        source_views = [_view_source_record(view) for view in target_views]
        # Context depth/conf identities are carried by their target-context copies only
        # after those copies intentionally drop depth. Replace the six records with
        # their full context records while retaining the 18 stable target identities.
        context_sources = {view.camera_name: _view_source_record(view) for view in context_views}
        for record in source_views[12:]:
            full = context_sources[record["camera_name"]]
            record["files"].update(
                {
                    key: value
                    for key, value in full["files"].items()
                    if key in {"depth_metric", "confidence"}
                }
            )

        artifact_paths = sorted(
            [temporary_dir / "transforms_train.json", temporary_dir / "transforms_test.json",
             temporary_dir / "points3D.ply"]
            + list(images_dir.glob("*.png")),
            key=lambda path: path.relative_to(temporary_dir).as_posix(),
        )
        numeric_payload = {
            "context_intrinsics": scene_data["context"]["intrinsics"].tolist(),
            "context_c2w": scene_data["context"]["c2w"].tolist(),
            "target_intrinsics": scene_data["target"]["intrinsics"].tolist(),
            "target_c2w": scene_data["target"]["c2w"].tolist(),
        }
        manifest: Dict[str, Any] = {
            "format_version": PREPARED_FORMAT_VERSION,
            "bin_token": bin_token,
            "split_index": split_index,
            "scene_name": scene_name,
            "resolution": list(resolution),
            "confidence_threshold": float(confidence_threshold),
            "split_sha256": scene_data["split_sha256"],
            "coordinate_convention": COORDINATE_CONVENTION,
            "camera_order": list(CAMERA_TYPES),
            "context_view_count": 6,
            "target_view_count": 18,
            "context_view_ids": scene_data["context"]["view_id"],
            "target_view_ids": scene_data["target"]["view_id"],
            "source_views": source_views,
            "bin_info": _file_identity(Path(scene_data["bin_info_path"])),
            "numeric_fingerprint": _canonical_sha256(numeric_payload),
            "point_cloud": point_summary,
            "code_fingerprint": _code_fingerprint(),
            "artifacts": [
                _relative_artifact_identity(path, temporary_dir) for path in artifact_paths
            ],
        }
        manifest["manifest_fingerprint"] = _canonical_sha256(manifest)
        _atomic_write_json(temporary_dir / "manifest.json", manifest)

        valid, reason = validate_prepared_scene(
            temporary_dir,
            bin_token,
            split_index,
            resolution,
            confidence_threshold,
            scene_data["split_sha256"],
        )
        if not valid:
            raise RuntimeError(f"Newly prepared scene failed validation: {reason}")

        backup = output_root / f".{scene_name}.old"
        if backup.exists():
            _safe_remove_directory(backup, output_root)
        if scene_dir.exists():
            os.replace(str(scene_dir), str(backup))
        os.replace(str(temporary_dir), str(scene_dir))
        if backup.exists():
            _safe_remove_directory(backup, output_root)
        return scene_dir
    finally:
        if temporary_dir.exists():
            _safe_remove_directory(temporary_dir, output_root)
