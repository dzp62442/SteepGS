#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def _read_omniscene_cameras(path, transformsfile):
    """Read prepared OmniScene cameras without Blender/OpenGL axis conversion."""
    transforms_path = os.path.join(path, transformsfile)
    with open(transforms_path, "r", encoding="utf-8") as transforms_file:
        contents = json.load(transforms_file)
    if contents.get("coordinate_convention") != "opencv_camera_to_keyframe_lidar_world":
        raise ValueError("Unsupported OmniScene coordinate convention in {}".format(transforms_path))
    if contents.get("no_flip_yz") is not True:
        raise ValueError("OmniScene transforms must explicitly set no_flip_yz=true")

    frames = contents.get("frames")
    if not isinstance(frames, list):
        raise ValueError("OmniScene transforms must contain a frame list: {}".format(transforms_path))
    cam_infos = []
    view_ids = []
    scene_root = os.path.realpath(path)
    for idx, frame in enumerate(frames):
        view_id = frame.get("view_id")
        if not isinstance(view_id, str) or not view_id:
            raise ValueError("OmniScene frame {} has no stable view_id".format(idx))
        view_ids.append(view_id)

        relative_path = frame.get("file_path")
        if not isinstance(relative_path, str):
            raise ValueError("OmniScene frame {} has no file_path".format(view_id))
        image_path = os.path.realpath(os.path.join(scene_root, relative_path))
        if os.path.commonpath([scene_root, image_path]) != scene_root:
            raise ValueError("OmniScene image path escapes scene root: {}".format(relative_path))
        if not os.path.isfile(image_path):
            raise FileNotFoundError("OmniScene image does not exist: {}".format(image_path))
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB").copy()
        width, height = image.size
        if frame.get("width") != width or frame.get("height") != height:
            raise ValueError("Image dimensions differ from frame metadata: {}".format(view_id))

        try:
            fx = float(frame["fl_x"])
            fy = float(frame["fl_y"])
            cx = float(frame["cx"])
            cy = float(frame["cy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid per-frame intrinsics: {}".format(view_id)) from exc
        if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
            raise ValueError("Non-finite or non-positive intrinsics: {}".format(view_id))
        if abs(cx - width / 2.0) > 1e-4 or abs(cy - height / 2.0) > 1e-4:
            raise ValueError(
                "SteepGS symmetric projection requires a centered principal point: {} "
                "has ({}, {}) for {}x{}".format(view_id, cx, cy, width, height)
            )

        c2w = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise ValueError("Invalid finite 4x4 c2w: {}".format(view_id))
        if not np.allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ValueError("Invalid homogeneous c2w bottom row: {}".format(view_id))
        rotation = c2w[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("Non-orthogonal c2w rotation: {}".format(view_id))
        if np.linalg.det(rotation) <= 0.0:
            raise ValueError("Non-positive c2w rotation determinant: {}".format(view_id))

        # The source pose already uses OpenCV camera axes.  Do not apply the
        # Blender reader's c2w[:3, 1:3] *= -1 conversion here.
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=focal2fov(fy, height),
                FovX=focal2fov(fx, width),
                image=image,
                image_path=image_path,
                image_name=view_id,
                width=width,
                height=height,
            )
        )
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("Duplicate OmniScene view_id in {}".format(transforms_path))
    return cam_infos


def readOmniSceneInfo(path, white_background, eval):
    """Load one strict 6-context/18-target prepared OmniScene scene."""
    del white_background
    if not eval:
        raise ValueError("OmniScene target views must remain evaluation-only; pass --eval")
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError("OmniScene manifest is required: {}".format(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("context_view_count") != 6 or manifest.get("target_view_count") != 18:
        raise ValueError("OmniScene manifest must declare exactly 6 train and 18 test views")

    print("Reading OmniScene Training Transforms")
    train_cam_infos = _read_omniscene_cameras(path, "transforms_train.json")
    print("Reading OmniScene Test Transforms")
    test_cam_infos = _read_omniscene_cameras(path, "transforms_test.json")
    if len(train_cam_infos) != 6 or len(test_cam_infos) != 18:
        raise ValueError(
            "OmniScene scene requires exactly 6 train and 18 test cameras; got {}/{}".format(
                len(train_cam_infos), len(test_cam_infos)
            )
        )
    expected_train_ids = manifest.get("context_view_ids")
    expected_test_ids = manifest.get("target_view_ids")
    if expected_train_ids != [camera.image_name for camera in train_cam_infos]:
        raise ValueError("Training view order differs from OmniScene manifest")
    if expected_test_ids != [camera.image_name for camera in test_cam_infos]:
        raise ValueError("Test view order differs from OmniScene manifest")

    ply_path = os.path.join(path, "points3D.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(
            "Metric-depth points3D.ply is required; random initialization is disabled: {}".format(
                ply_path
            )
        )
    pcd = fetchPly(ply_path)
    if pcd is None or len(pcd.points) == 0:
        raise ValueError("OmniScene initialization point cloud is empty: {}".format(ply_path))
    if not np.isfinite(pcd.points).all() or not np.isfinite(pcd.colors).all():
        raise ValueError("OmniScene initialization point cloud contains non-finite values")

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=getNerfppNorm(train_cam_infos),
        ply_path=ply_path,
    )

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "OmniScene": readOmniSceneInfo,
}
