"""Strict OmniScene loader used by the SteepGS preprocessing pipeline.

The loader deliberately keeps camera poses in the source OpenCV convention:
``sensor2lidar_transform`` maps camera coordinates to the key-frame LiDAR
coordinate system, which is used as the scene world coordinate system.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image


DATA_VERSION = "interp_12Hz_trainval"
DATASET_PREFIX = Path("/datasets/nuScenes")
CENTER150_FILENAME = "bins_center150_v1.json"
CENTER150_SAMPLE_COUNT = 150
CENTER150_TOKEN_PATTERN = re.compile(r"^scene([0-9a-f]+)_bin([0-9]+)$")
CAMERA_TYPES: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# Keep this list identical to DepthSplat/SVF-GS.  It is a compatibility/debug
# split only; Center150 remains the formal experiment split.
BINS_DEMO: Tuple[str, ...] = (
    "scenee7ef871f77f44331aefdebc24ec034b7_bin010",
    "scenee7ef871f77f44331aefdebc24ec034b7_bin200",
    "scene04219bfdc9004ba2af16d3079ecc4353_bin061",
    "scene07aed9dae37340a997535ad99138e243_bin058",
    "scene0ac05652a4c44374998be876ba5cd6fd_bin121",
    "scene16e50a63b809463099cb4c378fe0641e_bin231",
    "scene197a7e4d3de84e57af17b3d65fcb3893_bin177",
    "scene19d97841d6f64eba9f6eb9b6e8c257dc_bin001",
    "scene201b7c65a61f4bc1a2333ea90ba9a932_bin071",
    "scene2086743226764f268fe8d4b0b7c19590_bin043",
    "scene265f002f02d447ad9074813292eef75e_bin128",
    "scene26a6b03c8e2f4e6692f174a7074e54ff_bin103",
    "scene2abb3f3517c64446a5768df5665da49d_bin128",
    "scene2ca15f59d656489a8b1a0be4d9bead4e_bin003",
    "scene2ed0fcbfc214478ca3b3ce013e7723ba_bin154",
    "scene2f56eb47c64f43df8902d9f88aa8a019_bin136",
    "scene3045ed93c2534ec2a5cabea89b186bd9_bin176",
    "scene36f27b26ef4c423c9b79ac984dc33bae_bin207",
    "scene3a2d9bf6115f40898005d1c1df2b7282_bin107",
    "scene3ada261efee347cba2e7557794f1aec8_bin005",
    "scene3dd2be428534403ba150a0b60abc6a0a_bin083",
    "scene3dd9ad3f963e4f588d75c112cbf07f56_bin131",
    "scene3f90afe9f7dc49399347ae1626502aed_bin095",
    "scene4962cb207a824e57bd10a2af49354b16_bin089",
    "scene5301151d8b6a42b0b252e95634bd3995_bin121",
    "scene5521cd85ed0e441f8d23938ed09099dd_bin067",
    "scene6a24a80e2ea3493c81f5fdb9fe78c28a_bin033",
    "scene6af9b75e439e4811ad3b04dc2220657a_bin115",
    "scene7061c08f7eec4495979a0cf68ab6bb79_bin180",
    "scene7365495b74464629813b41eacdb711af_bin067",
    "scene76ceedbcc6a54b158eba9945a160a7bc_bin063",
    "scene7e8ff24069ff4023ac699669b2c920de_bin014",
    "scene813213458a214a39a1d1fc77fa52fa34_bin040",
    "scene848ac962547c4508b8f3b0fcc8d53270_bin023",
    "scene85651af9c04945c3a394cf845cb480a6_bin017",
    "scene8edbc31083ab4fb187626e5b3c0411f7_bin017",
    "scene9088db17416043e5880a53178bfa461c_bin005",
    "scene91c071bcc1ad4fa1b555399e1cfbab79_bin002",
    "scene91f797db8fb34ae5b32ba85eecae47c9_bin004",
    "scene9709626638f5406f9f773348150d02fd_bin092",
    "sceneafbc2583cc324938b2e8931d42c83e6b_bin009",
    "sceneb07358651c604e2d83da7c4d4755de73_bin017",
    "sceneb94fbf78579f4ff5ab5dbd897d5e3199_bin155",
    "scenecba3ddd5c3664a43b6a08e586e094900_bin032",
    "scened3b86ca0a17840109e9e049b3dd40037_bin040",
    "scenee036014a715945aa965f4ec24e8639c9_bin005",
    "sceneefa5c96f05594f41a2498eb9f2e7ad99_bin092",
    "scenef97bf749746c4c3a8ad9f1c11eab6444_bin009",
)


@dataclass
class OmniSceneView:
    view_id: str
    role: str
    camera_name: str
    frame_index: int
    image: torch.Tensor
    intrinsics: torch.Tensor
    c2w: torch.Tensor
    image_path: Path
    parameter_path: Path
    depth_path: Optional[Path] = None
    confidence_path: Optional[Path] = None
    depth_metric: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None


def resolve_data_root(value: Optional[Union[os.PathLike, str]] = None) -> Path:
    """Resolve the data root using CLI, environment, then workspace fallback."""
    if value is not None:
        candidate = Path(value).expanduser()
    elif os.environ.get("OMNISCENE_ROOT"):
        candidate = Path(os.environ["OMNISCENE_ROOT"]).expanduser()
    else:
        candidate = Path(__file__).resolve().parents[2] / "SVF-GS" / "data" / "nuScenes"
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"OmniScene data root does not exist: {candidate}")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_under_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Resolved path escapes OmniScene root: {resolved}") from exc
    return resolved


def _resolve_raw_path(raw_path: Union[os.PathLike, str], root: Path) -> Path:
    raw = Path(raw_path)
    try:
        relative = raw.relative_to(DATASET_PREFIX)
    except ValueError:
        if raw.is_absolute():
            try:
                relative = raw.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Source path is neither under {DATASET_PREFIX} nor {root}: {raw}"
                ) from exc
        else:
            relative = raw
    return _safe_under_root(root / relative, root)


def _replace_component(path: Path, source: Sequence[str], replacement: Sequence[str], root: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside OmniScene root: {path}") from exc
    parts = list(relative.parts)
    matches = [index for index, part in enumerate(parts) if part in source]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one path component in {tuple(source)}, found {len(matches)}: {path}"
        )
    index = matches[0]
    replacement_map = dict(zip(source, replacement))
    parts[index] = replacement_map[parts[index]]
    return _safe_under_root(root / Path(*parts), root)


def derive_condition_paths(raw_path: Union[os.PathLike, str], root: Path) -> Dict[str, Path]:
    """Apply the exact OmniScene samples/sweeps path transformation contract."""
    raw = _resolve_raw_path(raw_path, root)
    if raw.suffix.lower() != ".jpg":
        raise ValueError(f"OmniScene camera data_path must end in .jpg: {raw}")
    parameter = _replace_component(
        raw, ("samples", "sweeps"), ("samples_param_small", "sweeps_param_small"), root
    ).with_suffix(".json")
    image = _replace_component(
        raw, ("samples", "sweeps"), ("samples_small", "sweeps_small"), root
    )
    depth_base = _replace_component(
        image,
        ("samples_small", "sweeps_small"),
        ("samples_dptm_small", "sweeps_dptm_small"),
        root,
    )
    depth = depth_base.with_name(depth_base.stem + "_dpt.npy")
    confidence = depth_base.with_name(depth_base.stem + "_conf.npy")
    return {
        "raw": raw,
        "parameter": parameter,
        "image": image,
        "depth": depth,
        "confidence": confidence,
    }


def _validate_c2w(value: Any, label: str) -> np.ndarray:
    c2w = np.asarray(value, dtype=np.float64)
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError(f"{label}: c2w must be a finite 4x4 matrix")
    if not np.allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{label}: invalid homogeneous c2w bottom row")
    rotation = c2w[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{label}: c2w rotation is not orthogonal")
    if not np.linalg.det(rotation) > 0.0:
        raise ValueError(f"{label}: c2w rotation has non-positive determinant")
    return c2w.astype(np.float32)


def _resize_float_map(array: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = size_hw
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D depth/confidence map, got {array.shape}")
    if array.shape == (target_h, target_w):
        return array.astype(np.float32, copy=True)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(
        Image.fromarray(array.astype(np.float32)).resize((target_w, target_h), resampling),
        dtype=np.float32,
    ).copy()


def _load_view(
    info: Dict[str, Any],
    root: Path,
    resolution: Tuple[int, int],
    view_id: str,
    role: str,
    camera_name: str,
    frame_index: int,
    load_depth_confidence: bool,
) -> OmniSceneView:
    paths = derive_condition_paths(info["data_path"], root)
    required = [paths["image"], paths["parameter"]]
    if load_depth_confidence:
        required.extend([paths["depth"], paths["confidence"]])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing OmniScene condition files: " + ", ".join(missing))

    with paths["parameter"].open("r", encoding="utf-8") as parameter_file:
        parameter = json.load(parameter_file)
    intrinsic = np.asarray(parameter["camera_intrinsic"], dtype=np.float32)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"{paths['parameter']}: camera_intrinsic must be a finite 3x3 matrix")
    if not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-6) or not np.allclose(
        [intrinsic[0, 1], intrinsic[1, 0]], [0.0, 0.0], atol=1e-6
    ):
        raise ValueError(f"{paths['parameter']}: skewed/projective intrinsics are unsupported")

    target_h, target_w = resolution
    with Image.open(paths["image"]) as source_image:
        source_image = source_image.convert("RGB")
        source_w, source_h = source_image.size
        if (source_h, source_w) != resolution:
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            source_image = source_image.resize((target_w, target_h), resampling)
        image_array = np.asarray(source_image, dtype=np.uint8).copy()

    scale_w = target_w / source_w
    scale_h = target_h / source_h
    intrinsic = intrinsic.copy()
    intrinsic[0, 0] *= scale_w
    intrinsic[0, 2] *= scale_w
    intrinsic[1, 1] *= scale_h
    intrinsic[1, 2] *= scale_h
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError(f"{paths['parameter']}: focal lengths must be positive")
    if abs(float(intrinsic[0, 2]) - target_w / 2.0) > 1e-4 or abs(
        float(intrinsic[1, 2]) - target_h / 2.0
    ) > 1e-4:
        raise ValueError(
            f"{paths['parameter']}: SteepGS requires a centered principal point after resize"
        )

    depth_tensor = None
    confidence_tensor = None
    if load_depth_confidence:
        depth = _resize_float_map(np.load(paths["depth"]), resolution)
        confidence = _resize_float_map(np.load(paths["confidence"]), resolution)
        if depth.shape != resolution or confidence.shape != resolution:
            raise ValueError(f"Depth/confidence resize failed for {view_id}")
        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth)).float()
        confidence_tensor = torch.from_numpy(np.ascontiguousarray(confidence)).float()

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float().div_(255.0)
    c2w = _validate_c2w(info["sensor2lidar_transform"], view_id)
    return OmniSceneView(
        view_id=view_id,
        role=role,
        camera_name=camera_name,
        frame_index=frame_index,
        image=image_tensor,
        intrinsics=torch.from_numpy(intrinsic),
        c2w=torch.from_numpy(c2w),
        image_path=paths["image"],
        parameter_path=paths["parameter"],
        depth_path=paths["depth"] if load_depth_confidence else None,
        confidence_path=paths["confidence"] if load_depth_confidence else None,
        depth_metric=depth_tensor,
        confidence=confidence_tensor,
    )


class OmniSceneDataset:
    """Map-style dataset with strict view counts and stable view identities."""

    def __init__(
        self,
        data_root: Optional[Union[os.PathLike, str]] = None,
        mode: str = "center150",
        resolution: Tuple[int, int] = (112, 200),
    ) -> None:
        self.data_root = resolve_data_root(data_root)
        self.mode = mode
        self.resolution = (int(resolution[0]), int(resolution[1]))
        if self.resolution not in ((112, 200), (224, 400)):
            raise ValueError("Supported resolutions are exactly 112x200 and 224x400")
        self.version_root = self.data_root / DATA_VERSION
        if not self.version_root.is_dir():
            raise FileNotFoundError(f"Missing OmniScene data version: {self.version_root}")

        self.split_path: Optional[Path]
        if mode == "center150":
            self.split_path = self.version_root / CENTER150_FILENAME
            self.bin_tokens = self._load_center150()
        elif mode in {"train", "val", "test"}:
            split_name = "train" if mode == "train" else "val"
            self.split_path = self.version_root / f"bins_{split_name}_3.2m.json"
            tokens = self._load_bins_file(self.split_path)
            if mode == "val":
                tokens = tokens[:30000:3000][:10]
            elif mode == "test":
                tokens = tokens[0::14][:2048]
            self.bin_tokens = tokens
            self._validate_bin_info_files(tokens)
        elif mode == "demo":
            self.split_path = None
            self.bin_tokens = list(BINS_DEMO)
            self._validate_bin_info_files(self.bin_tokens)
        else:
            raise ValueError(f"Unsupported OmniScene mode: {mode}")

        if self.split_path is not None:
            self.split_sha256 = sha256_file(self.split_path)
        else:
            encoded = json.dumps(self.bin_tokens, separators=(",", ":")).encode("utf-8")
            self.split_sha256 = hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _load_bins_file(path: Path) -> List[str]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing OmniScene split: {path}")
        with path.open("r", encoding="utf-8") as split_file:
            payload = json.load(split_file)
        tokens = payload.get("bins") if isinstance(payload, dict) else None
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError(f"Split must contain a string list named 'bins': {path}")
        return tokens

    def _validate_bin_info_files(self, tokens: Sequence[str]) -> None:
        for token in tokens:
            if CENTER150_TOKEN_PATTERN.fullmatch(token) is None:
                raise ValueError(f"Invalid OmniScene bin token: {token}")
            info_path = self.version_root / "bin_infos_3.2m" / f"{token}.pkl"
            if not info_path.is_file():
                raise FileNotFoundError(f"Missing OmniScene bin metadata: {info_path}")

    def _load_center150(self) -> List[str]:
        tokens = self._load_bins_file(self.split_path)
        if len(tokens) != CENTER150_SAMPLE_COUNT or len(set(tokens)) != CENTER150_SAMPLE_COUNT:
            raise ValueError(
                f"Center150 must contain exactly {CENTER150_SAMPLE_COUNT} unique bins; "
                f"got {len(tokens)} entries and {len(set(tokens))} unique entries"
            )
        scene_tokens = []
        for token in tokens:
            match = CENTER150_TOKEN_PATTERN.fullmatch(token)
            if match is None:
                raise ValueError(f"Invalid Center150 bin token: {token}")
            scene_tokens.append(match.group(1))
        if len(set(scene_tokens)) != CENTER150_SAMPLE_COUNT:
            raise ValueError(
                f"Center150 must cover {CENTER150_SAMPLE_COUNT} unique scenes; "
                f"got {len(set(scene_tokens))}"
            )
        self._validate_bin_info_files(tokens)
        return tokens

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def bin_info_path(self, index: int) -> Path:
        return self.version_root / "bin_infos_3.2m" / f"{self.bin_tokens[index]}.pkl"

    def __getitem__(self, index: int) -> Dict[str, Any]:
        bin_token = self.bin_tokens[index]
        info_path = self.bin_info_path(index)
        with info_path.open("rb") as info_file:
            bin_info = pickle.load(info_file)
        sensor_info = bin_info.get("sensor_info")
        if not isinstance(sensor_info, dict):
            raise ValueError(f"{info_path}: missing sensor_info mapping")

        context_views: List[OmniSceneView] = []
        novel_views: List[OmniSceneView] = []
        for camera_name in CAMERA_TYPES:
            camera_frames = sensor_info.get(camera_name)
            if not isinstance(camera_frames, Sequence) or len(camera_frames) < 3:
                raise ValueError(f"{bin_token}/{camera_name}: at least three frames are required")
            context_views.append(
                _load_view(
                    camera_frames[0], self.data_root, self.resolution,
                    f"context_{len(context_views):02d}_{camera_name}", "context",
                    camera_name, 0, True,
                )
            )
            for frame_index in (1, 2):
                target_index = len(novel_views)
                novel_views.append(
                    _load_view(
                        camera_frames[frame_index], self.data_root, self.resolution,
                        f"target_{target_index:02d}_{camera_name}_t{frame_index}", "novel",
                        camera_name, frame_index, False,
                    )
                )

        target_context_views = [
            replace(
                view,
                view_id=f"target_{12 + camera_index:02d}_{view.camera_name}_context",
                role="target_context",
                depth_path=None,
                confidence_path=None,
                depth_metric=None,
                confidence=None,
            )
            for camera_index, view in enumerate(context_views)
        ]
        target_views = novel_views + target_context_views
        context_ids = [view.view_id for view in context_views]
        target_ids = [view.view_id for view in target_views]
        if len(context_views) != 6 or len(set(context_ids)) != 6:
            raise RuntimeError(f"{bin_token}: expected six unique context views")
        if len(target_views) != 18 or len(set(target_ids)) != 18:
            raise RuntimeError(f"{bin_token}: expected eighteen unique target views")

        def stack(views: Sequence[OmniSceneView], name: str) -> torch.Tensor:
            return torch.stack([getattr(view, name) for view in views], dim=0)

        return {
            "scene": bin_token,
            "index": index,
            "split_sha256": self.split_sha256,
            "bin_info_path": info_path,
            "context": {
                "views": context_views,
                "view_id": context_ids,
                "image": stack(context_views, "image"),
                "intrinsics": stack(context_views, "intrinsics"),
                "c2w": stack(context_views, "c2w"),
                "depth_metric": stack(context_views, "depth_metric"),
                "confidence": stack(context_views, "confidence"),
            },
            "target": {
                "views": target_views,
                "view_id": target_ids,
                "image": stack(target_views, "image"),
                "intrinsics": stack(target_views, "intrinsics"),
                "c2w": stack(target_views, "c2w"),
            },
        }
