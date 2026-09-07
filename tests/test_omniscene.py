import argparse
import json
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from comp_svfgs.dataset_omniscene import (
    BINS_DEMO,
    CAMERA_TYPES,
    CENTER150_SAMPLE_COUNT,
    DATA_VERSION,
    OmniSceneDataset,
    derive_condition_paths,
)
from comp_svfgs.omniscene_preprocess import (
    build_initial_point_cloud,
    preprocess_scene,
    validate_prepared_scene,
)
from scene.colmap_loader import Camera as ColmapCamera
from scene.colmap_loader import Image as ColmapImage
from scene.dataset_readers import (
    _read_omniscene_cameras,
    readCamerasFromTransforms,
    readColmapCameras,
    readOmniSceneInfo,
)
from scripts.run_omniscene import (
    _ensure_result_protocol,
    _prepare_if_needed,
    _refresh_scene_metric_reports,
    _safe_remove_scene_directory,
    _scene_protocol,
    _write_completion,
    aggregate_center150_results,
    build_train_command,
    validate_extra_train_args,
    validate_scene_result,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_center150_root(tmp_path: Path) -> Path:
    root = tmp_path / "nuScenes"
    version_root = root / DATA_VERSION
    info_root = version_root / "bin_infos_3.2m"
    info_root.mkdir(parents=True)
    tokens = [f"scene{index + 1:032x}_bin000" for index in range(CENTER150_SAMPLE_COUNT)]
    _write_json(version_root / "bins_center150_v1.json", {"bins": tokens})

    sensor_info = {}
    for camera_index, camera_name in enumerate(CAMERA_TYPES):
        frames = []
        for frame_index in range(3):
            filename = f"{camera_name}_{frame_index}.jpg"
            raw_path = f"/datasets/nuScenes/samples/{camera_name}/{filename}"
            frames.append(
                {
                    "data_path": raw_path,
                    "sensor2lidar_transform": np.array(
                        [
                            [1.0, 0.0, 0.0, float(camera_index)],
                            [0.0, 1.0, 0.0, float(frame_index) * 0.1],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                }
            )
            paths = derive_condition_paths(raw_path, root.resolve())
            paths["image"].parent.mkdir(parents=True, exist_ok=True)
            image = np.zeros((112, 200, 3), dtype=np.uint8)
            image[..., camera_index % 3] = 30 + 20 * frame_index
            Image.fromarray(image, mode="RGB").save(paths["image"], format="JPEG")
            _write_json(
                paths["parameter"],
                {"camera_intrinsic": [[100.0, 0.0, 100.0], [0.0, 110.0, 56.0], [0.0, 0.0, 1.0]]},
            )
            paths["depth"].parent.mkdir(parents=True, exist_ok=True)
            depth = np.ones((112, 200), dtype=np.float32) * (10.0 + camera_index)
            confidence = np.zeros((112, 200), dtype=np.float32)
            confidence[20 + camera_index, 30 + frame_index] = 0.8
            np.save(paths["depth"], depth)
            np.save(paths["confidence"], confidence)
        sensor_info[camera_name] = frames
    payload = {"sensor_info": sensor_info}
    for token in tokens:
        with (info_root / f"{token}.pkl").open("wb") as output_file:
            pickle.dump(payload, output_file)
    return root


def _case_center150_loader_contract_and_view_order(tmp_path):
    root = _make_center150_root(tmp_path)
    dataset = OmniSceneDataset(root, mode="center150", resolution=(112, 200))
    assert len(dataset) == 150
    sample = dataset[0]
    assert sample["context"]["image"].shape == (6, 3, 112, 200)
    assert sample["context"]["depth_metric"].shape == (6, 112, 200)
    assert sample["target"]["image"].shape == (18, 3, 112, 200)
    assert sample["target"]["view_id"][:4] == [
        "target_00_CAM_FRONT_t1",
        "target_01_CAM_FRONT_t2",
        "target_02_CAM_FRONT_RIGHT_t1",
        "target_03_CAM_FRONT_RIGHT_t2",
    ]
    assert sample["target"]["view_id"][-1] == "target_17_CAM_BACK_RIGHT_context"
    assert sample["target"]["views"][-1].depth_metric is None
    assert torch.equal(sample["target"]["image"][-1], sample["context"]["image"][-1])
    assert sample["context"]["intrinsics"][0, 0, 0].item() == 100.0
    assert sample["context"]["intrinsics"][0, 0, 2].item() == 100.0


def _case_center150_rejects_duplicate_scene(tmp_path):
    root = _make_center150_root(tmp_path)
    split_path = root / DATA_VERSION / "bins_center150_v1.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["bins"][1] = payload["bins"][0].replace("bin000", "bin001")
    _write_json(split_path, payload)
    with unittest.TestCase().assertRaisesRegex(ValueError, "unique scenes"):
        OmniSceneDataset(root, mode="center150")


def _case_compatible_mode_selection(tmp_path):
    root = _make_center150_root(tmp_path)
    version_root = root / DATA_VERSION
    center_tokens = json.loads(
        (version_root / "bins_center150_v1.json").read_text(encoding="utf-8")
    )["bins"]
    _write_json(version_root / "bins_train_3.2m.json", {"bins": center_tokens[:3]})
    _write_json(version_root / "bins_val_3.2m.json", {"bins": center_tokens})
    assert OmniSceneDataset(root, mode="train").bin_tokens == center_tokens[:3]
    assert OmniSceneDataset(root, mode="val").bin_tokens == center_tokens[:30000:3000][:10]
    assert OmniSceneDataset(root, mode="test").bin_tokens == center_tokens[0::14][:2048]

    info_root = version_root / "bin_infos_3.2m"
    for token in BINS_DEMO:
        info_path = info_root / f"{token}.pkl"
        if not info_path.exists():
            with info_path.open("wb") as output_file:
                pickle.dump({}, output_file)
    assert OmniSceneDataset(root, mode="demo").bin_tokens == list(BINS_DEMO)


def _case_path_conversion_is_component_scoped_and_root_bounded(tmp_path):
    root = tmp_path.resolve()
    paths = derive_condition_paths(
        "/datasets/nuScenes/sweeps/CAM_FRONT/example.jpg", root
    )
    assert paths["image"] == root / "sweeps_small/CAM_FRONT/example.jpg"
    assert paths["parameter"] == root / "sweeps_param_small/CAM_FRONT/example.json"
    assert paths["depth"] == root / "sweeps_dptm_small/CAM_FRONT/example_dpt.npy"
    with unittest.TestCase().assertRaises(ValueError):
        derive_condition_paths("/outside/samples/CAM_FRONT/example.jpg", root)


def _case_backprojection_and_reprojection_are_opencv_consistent():
    height, width = 3, 4
    images = torch.zeros((6, 3, height, width), dtype=torch.float32)
    images[:, 0, 1, 2] = 1.0
    depth = torch.zeros((6, height, width), dtype=torch.float32)
    depth[:, 1, 2] = 5.0
    confidence = torch.zeros_like(depth)
    confidence[:, 1, 2] = 0.30001
    intrinsics = torch.tensor(
        [[[2.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]] * 6
    )
    c2w = torch.eye(4).repeat(6, 1, 1)
    context = {
        "image": images,
        "depth_metric": depth,
        "confidence": confidence,
        "intrinsics": intrinsics,
        "c2w": c2w,
    }
    points, colors, summary = build_initial_point_cloud(context, 0.3)
    assert points.shape == (6, 3)
    assert np.allclose(points[0], [2.5, 0.0, 5.0])
    projected_u = points[0, 0] * 2.0 / points[0, 2] + 1.0
    projected_v = points[0, 1] * 4.0 / points[0, 2] + 1.0
    assert abs(projected_u - 2.0) < 1e-4
    assert abs(projected_v - 1.0) < 1e-4
    assert colors[0].tolist() == [255, 0, 0]
    assert summary["point_count"] == 6


def _case_preprocess_cache_and_strict_reader(tmp_path):
    root = _make_center150_root(tmp_path)
    dataset = OmniSceneDataset(root, mode="center150", resolution=(112, 200))
    sample = dataset[0]
    prepared_root = tmp_path / "prepared"
    scene_dir = preprocess_scene(sample, prepared_root, 0.3)
    assert scene_dir.name.startswith("001_")
    train = json.loads((scene_dir / "transforms_train.json").read_text(encoding="utf-8"))
    test = json.loads((scene_dir / "transforms_test.json").read_text(encoding="utf-8"))
    assert len(train["frames"]) == 6
    assert len(test["frames"]) == 18
    assert len({frame["file_path"] for frame in train["frames"] + test["frames"]}) == 18
    assert test["frames"][12]["file_path"] == train["frames"][0]["file_path"]
    valid, reason = validate_prepared_scene(
        scene_dir, sample["scene"], 0, (112, 200), 0.3, dataset.split_sha256
    )
    assert valid, reason
    vertices = PlyData.read(str(scene_dir / "points3D.ply"))["vertex"]
    assert len(vertices) == 6
    cameras = _read_omniscene_cameras(scene_dir, "transforms_test.json")
    assert [camera.image_name for camera in cameras] == sample["target"]["view_id"]
    scene_info = readOmniSceneInfo(scene_dir, False, True)
    assert len(scene_info.train_cameras) == 6
    assert len(scene_info.test_cameras) == 18
    assert np.isfinite(scene_info.point_cloud.points).all()

    # Source metadata changes with identical content still validates by SHA fallback.
    source_image = sample["context"]["views"][0].image_path
    os.utime(source_image, None)
    valid, reason = validate_prepared_scene(
        scene_dir, sample["scene"], 0, (112, 200), 0.3, dataset.split_sha256
    )
    assert valid, reason


def _case_reader_rejects_off_center_principal_point(tmp_path):
    root = _make_center150_root(tmp_path)
    dataset = OmniSceneDataset(root, mode="center150", resolution=(112, 200))
    scene_dir = preprocess_scene(dataset[0], tmp_path / "prepared", 0.3)
    transform_path = scene_dir / "transforms_test.json"
    payload = json.loads(transform_path.read_text(encoding="utf-8"))
    payload["frames"][0]["cx"] += 0.01
    _write_json(transform_path, payload)
    with unittest.TestCase().assertRaisesRegex(ValueError, "centered principal point"):
        _read_omniscene_cameras(scene_dir, "transforms_test.json")


def _case_train_command_is_strict_and_checkpoint_free(tmp_path):
    command = build_train_command(
        tmp_path / "scene", tmp_path / "result", 1000, (1000,), 0, ("--quiet",)
    )
    assert command[0] == os.sys.executable
    assert command[command.index("-r") + 1] == "1"
    assert command[command.index("--densify_strategy") + 1] == "steepest"
    assert command[command.index("--S_estimator") + 1] == "inv_cov"
    assert "--omniscene_protocol" in command
    assert not any("checkpoint" in value for value in command)

    parser = argparse.ArgumentParser()
    with unittest.TestCase().assertRaises(SystemExit):
        validate_extra_train_args(parser, ["--iterations", "2000"])


def _case_prepared_cache_reuse_avoids_raw_loading(tmp_path):
    root = _make_center150_root(tmp_path)
    dataset = OmniSceneDataset(root, mode="center150", resolution=(112, 200))
    prepared_root = tmp_path / "prepared"
    scene_dir = preprocess_scene(dataset[0], prepared_root, 0.3)
    with mock.patch.object(OmniSceneDataset, "__getitem__", side_effect=AssertionError("raw load")):
        reused = _prepare_if_needed(dataset, 0, scene_dir, prepared_root, 0.3)
    assert reused == scene_dir


def _case_protocol_conflict_and_safe_cleanup(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    semantics = {
        "protocol_version": 1,
        "mode": "center150",
        "data_version": DATA_VERSION,
        "split_sha256": "split",
        "sample_count": 150,
        "resolution": [112, 200],
        "confidence_threshold": 0.3,
        "train_view_count": 6,
        "target_view_count": 18,
        "iterations": 10000,
        "eval_iterations": [1000, 5000, 10000],
        "seed": 0,
        "state_semantics": "before update",
        "metric_protocol": {"primary": "all_18"},
        "training_command_contract": {"checkpointing": False},
        "effective_training_parameters": {"optimization": {"iterations": 10000}},
        "determinism": {"seed": 0},
    }
    protocol = dict(semantics, fingerprint="one", git={"commit": "old"})
    _ensure_result_protocol(protocol_path, protocol)
    _ensure_result_protocol(protocol_path, protocol)
    _ensure_result_protocol(
        protocol_path,
        dict(
            semantics,
            fingerprint="two",
            git={"commit": "new"},
            code_sha256={"scripts/run_omniscene.py": "changed"},
            software={"torch": "new"},
        ),
    )
    with unittest.TestCase().assertRaises(RuntimeError):
        _ensure_result_protocol(
            protocol_path,
            dict(semantics, fingerprint="three", resolution=[224, 400]),
        )

    experiment_root = tmp_path / "results"
    scene_dir = experiment_root / "001_scene"
    scene_dir.mkdir(parents=True)
    _safe_remove_scene_directory(scene_dir, experiment_root, ["001_scene"])
    assert not scene_dir.exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    with unittest.TestCase().assertRaises(ValueError):
        _safe_remove_scene_directory(outside, experiment_root, ["outside"])


def _case_existing_colmap_and_blender_readers(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (8, 4), color=(1, 2, 3)).save(images_dir / "camera.png")
    intrinsics = {
        1: ColmapCamera(
            id=1, model="PINHOLE", width=8, height=4,
            params=np.array([4.0, 5.0, 4.0, 2.0]),
        )
    }
    extrinsics = {
        1: ColmapImage(
            id=1, qvec=np.array([1.0, 0.0, 0.0, 0.0]), tvec=np.zeros(3),
            camera_id=1, name="camera.png", xys=None, point3D_ids=None,
        )
    }
    colmap_cameras = readColmapCameras(extrinsics, intrinsics, str(images_dir))
    assert len(colmap_cameras) == 1
    assert colmap_cameras[0].image_name == "camera"
    assert np.allclose(colmap_cameras[0].R, np.eye(3))
    colmap_cameras[0].image.close()

    Image.new("RGB", (8, 4), color=(4, 5, 6)).save(tmp_path / "blender.png")
    _write_json(
        tmp_path / "transforms_train.json",
        {"camera_angle_x": 1.0, "frames": [{"file_path": "blender", "transform_matrix": np.eye(4).tolist()}]},
    )
    blender_cameras = readCamerasFromTransforms(
        str(tmp_path), "transforms_train.json", False, extension=".png"
    )
    assert len(blender_cameras) == 1
    assert np.allclose(blender_cameras[0].R, np.diag([1.0, -1.0, -1.0]))
    blender_cameras[0].image.close()


def _write_fake_gaussian_ply(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    vertices = np.zeros(2, dtype=dtype)
    vertices["z"] = 1.0
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def _case_strict_result_completion(tmp_path):
    root = _make_center150_root(tmp_path)
    dataset = OmniSceneDataset(root, mode="center150", resolution=(112, 200))
    scene_dir = preprocess_scene(dataset[0], tmp_path / "prepared", 0.3)
    model_path = tmp_path / "result"
    model_path.mkdir()
    scene_protocol = _scene_protocol(
        {"fingerprint": "global-before-commit"},
        scene_dir,
        scene_dir.name,
        dataset.bin_tokens[0],
    )
    _write_json(model_path / "protocol.json", scene_protocol)
    (model_path / "cfg_args").write_text("Namespace()", encoding="utf-8")
    (model_path / "train_log.txt").write_text("complete\n", encoding="utf-8")
    _write_json(
        model_path / "omniscene_training_trace.json",
        {
            "iterations": 2,
            "selection_count": 2,
            "view_sequence_sha256": "0" * 64,
            "gaussian_count_at_milestones": {"1": 2, "2": 2},
            "gaussian_count_after_loop": 2,
        },
    )
    view_ids = json.loads(
        (scene_dir / "transforms_test.json").read_text(encoding="utf-8")
    )["frames"]
    view_ids = [frame["view_id"] for frame in view_ids]
    for iteration, training_time in ((1, 0.5), (2, 1.0)):
        records = [
            {
                "target_index": index,
                "view_id": view_id,
                "role": "novel" if index < 12 else "context",
                "psnr": 20.0 + index / 100.0,
                "ssim": 0.8,
                "lpips": 0.2,
            }
            for index, view_id in enumerate(view_ids)
        ]
        all_metrics = {
            name: sum(record[name] for record in records) / 18 for name in ("psnr", "ssim", "lpips")
        }
        novel_metrics = {
            name: sum(record[name] for record in records[:12]) / 12 for name in ("psnr", "ssim", "lpips")
        }
        _write_json(
            model_path / f"metrics_{iteration}.json",
            {
                "iteration": iteration,
                "target_view_count": 18,
                "training_time_seconds": training_time,
                "all_18": all_metrics,
                "novel_12": novel_metrics,
                "per_view": records,
            },
        )
        (model_path / f"metrics_{iteration}.txt").write_text(
            "PSNR: {:.10f}\nSSIM: {:.10f}\nLPIPS: {:.10f}\n".format(
                all_metrics["psnr"], all_metrics["ssim"], all_metrics["lpips"]
            ),
            encoding="utf-8",
        )
        (model_path / f"training_time_{iteration}.txt").write_text(
            f"TRAINING_TIME_SECONDS: {training_time:.10f}\n", encoding="utf-8"
        )
        for directory_name in ("renders", "gt"):
            directory = model_path / "test" / f"ours_{iteration}" / directory_name
            directory.mkdir(parents=True)
            for view_id in view_ids:
                Image.new("RGB", (200, 112), color=(10, 20, 30)).save(directory / f"{view_id}.png")
    _write_fake_gaussian_ply(
        model_path / "point_cloud" / "iteration_2" / "point_cloud.ply"
    )
    arguments = (
        model_path, scene_dir, dataset.bin_tokens[0], 0, (112, 200), 0.3,
        dataset.split_sha256, 2, (1, 2), scene_protocol,
    )
    valid, reason = validate_scene_result(*arguments, require_completion=False)
    assert valid, reason
    _write_completion(model_path, scene_protocol, 2, (1, 2))
    valid, reason = validate_scene_result(*arguments, require_completion=True)
    assert valid, reason
    resumed_protocol = dict(scene_protocol)
    resumed_protocol["global_protocol_fingerprint"] = "global-after-commit"
    resumed_protocol["fingerprint"] = "scene-after-commit"
    resumed_arguments = (*arguments[:-1], resumed_protocol)
    valid, reason = validate_scene_result(*resumed_arguments, require_completion=True)
    assert valid, reason
    timing_paths = [model_path / f"training_time_{iteration}.txt" for iteration in (1, 2)]
    timing_snapshots = [
        (path.read_bytes(), path.stat().st_mtime_ns) for path in timing_paths
    ]
    assert _refresh_scene_metric_reports(model_path, scene_dir, (1, 2)) == [1, 2]
    assert "[ALL_18]" in (model_path / "metrics_1.txt").read_text(encoding="utf-8")
    assert "[NOVEL_12]" in (model_path / "metrics_1.txt").read_text(encoding="utf-8")
    assert _refresh_scene_metric_reports(model_path, scene_dir, (1, 2)) == []
    assert [
        (path.read_bytes(), path.stat().st_mtime_ns) for path in timing_paths
    ] == timing_snapshots
    valid, reason = validate_scene_result(*resumed_arguments, require_completion=True)
    assert valid, reason
    incompatible_protocol = dict(resumed_protocol, bin_token="different-bin")
    valid, _ = validate_scene_result(
        *arguments[:-1], incompatible_protocol, require_completion=True
    )
    assert not valid
    (model_path / "test/ours_1/renders" / f"{view_ids[0]}.png").unlink()
    valid, _ = validate_scene_result(*resumed_arguments, require_completion=True)
    assert not valid


def _case_center150_aggregation_is_macro_mean(tmp_path):
    records = [
        (f"{index + 1:03d}_token", "token", tmp_path / "scene", tmp_path / f"result-{index}")
        for index in range(CENTER150_SAMPLE_COUNT)
    ]
    dataset = SimpleNamespace(split_sha256="split")
    protocol = {"fingerprint": "global", "resolution": [112, 200]}
    fake_metrics = {
        "all_18": {"psnr": 20.0, "ssim": 0.8, "lpips": 0.2},
        "novel_12": {"psnr": 19.0, "ssim": 0.7, "lpips": 0.3},
        "training_time_seconds": 10.0,
        "per_view": [],
    }
    with mock.patch("scripts.run_omniscene._scene_protocol", return_value={"fingerprint": "scene"}), mock.patch(
        "scripts.run_omniscene.validate_scene_result", return_value=(True, "complete")
    ), mock.patch("scripts.run_omniscene._read_iteration_result", return_value=fake_metrics):
        summary = aggregate_center150_results(
            tmp_path, records, dataset, protocol, 0.3, 1, (1,)
        )
    assert summary["sample_count"] == 150
    assert summary["milestones"]["1"]["all_18"]["psnr"] == {"mean": 20.0, "std": 0.0}
    assert summary["milestones"]["1"]["novel_12"]["psnr"] == {"mean": 19.0, "std": 0.0}
    assert len(summary["samples"]) == 150
    summary_text = (tmp_path / "center150_metrics_summary.txt").read_text(encoding="utf-8")
    assert "[ALL_18]" in summary_text
    assert "[NOVEL_12]" in summary_text


class OmniSceneTests(unittest.TestCase):
    def _with_tmp_path(self, function):
        with tempfile.TemporaryDirectory() as temporary:
            function(Path(temporary))

    def test_center150_loader_contract_and_view_order(self):
        self._with_tmp_path(_case_center150_loader_contract_and_view_order)

    def test_center150_rejects_duplicate_scene(self):
        self._with_tmp_path(_case_center150_rejects_duplicate_scene)

    def test_compatible_mode_selection(self):
        self._with_tmp_path(_case_compatible_mode_selection)

    def test_path_conversion_is_component_scoped_and_root_bounded(self):
        self._with_tmp_path(_case_path_conversion_is_component_scoped_and_root_bounded)

    def test_backprojection_and_reprojection_are_opencv_consistent(self):
        _case_backprojection_and_reprojection_are_opencv_consistent()

    def test_preprocess_cache_and_strict_reader(self):
        self._with_tmp_path(_case_preprocess_cache_and_strict_reader)

    def test_reader_rejects_off_center_principal_point(self):
        self._with_tmp_path(_case_reader_rejects_off_center_principal_point)

    def test_train_command_is_strict_and_checkpoint_free(self):
        self._with_tmp_path(_case_train_command_is_strict_and_checkpoint_free)

    def test_prepared_cache_reuse_avoids_raw_loading(self):
        self._with_tmp_path(_case_prepared_cache_reuse_avoids_raw_loading)

    def test_protocol_conflict_and_safe_cleanup(self):
        self._with_tmp_path(_case_protocol_conflict_and_safe_cleanup)

    def test_existing_colmap_and_blender_readers(self):
        self._with_tmp_path(_case_existing_colmap_and_blender_readers)

    def test_strict_result_completion(self):
        self._with_tmp_path(_case_strict_result_completion)

    def test_center150_aggregation_is_macro_mean(self):
        self._with_tmp_path(_case_center150_aggregation_is_macro_mean)


if __name__ == "__main__":
    unittest.main()
