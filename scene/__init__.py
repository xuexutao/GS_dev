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
import random
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from utils.graphics_utils import geom_transform_points
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        mask_opt=None,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # Keep dataset/mask context for optional per-object export.
        self._source_path = getattr(args, "source_path", None)
        self._images_subdir = "images" if getattr(args, "images", None) is None else str(getattr(args, "images"))
        self._mask_opt = mask_opt

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.depths, args.eval, args.train_test_exp
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # Optional: object-only reconstruction.
        # If mask_only is enabled and a specific mask_object_id is provided, filter COLMAP points3D
        # so the initial Gaussians come only from the selected object.
        try:
            if (
                mask_opt is not None
                and bool(getattr(mask_opt, "mask_only", False))
                and int(getattr(mask_opt, "mask_object_id", -1)) >= 0
            ):
                from scene.dataset_readers import build_object_pointcloud_from_colmap_masks

                images_subdir = "images" if args.images is None else args.images
                filtered_pcd = build_object_pointcloud_from_colmap_masks(
                    source_path=args.source_path,
                    images_subdir=images_subdir,
                    mask_dirname=str(getattr(mask_opt, "mask_dirname", "masks_sam")),
                    object_id=int(getattr(mask_opt, "mask_object_id", 0)),
                    sparse_subdir=os.path.join("sparse", "0"),
                    min_inlier_ratio=float(getattr(mask_opt, "mask_points3d_min_ratio", 0.5)),
                    min_inlier_count=int(getattr(mask_opt, "mask_points3d_min_count", 1)),
                )
                scene_info = scene_info._replace(point_cloud=filtered_pcd)
                print(
                    f"[MaskOnly] Filtered COLMAP points3D with obj_{int(getattr(mask_opt, 'mask_object_id', 0)):04d}.png; "
                    f"initial points = {filtered_pcd.points.shape[0]}"
                )
        except Exception as e:
            print(f"[WARN] Failed to filter COLMAP points3D for mask_only: {e}. Fallback to full point cloud.")

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False, args.start_type)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True, args.start_type)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        # Optional: export per-object gaussian point clouds if SAM masks exist.
        try:
            self._export_object_gaussians(point_cloud_path)
        except Exception as e:
            print(f"[WARN] Failed to export per-object gaussian ply: {e}")
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def _export_object_gaussians(self, point_cloud_path: str) -> None:
        """基于 2D masks，把最终高斯粗分配到各个 obj，并导出 obj_XXXX.ply。"""

        if self._mask_opt is None:
            return

        if not self._source_path:
            return

        mask_dirname = str(getattr(self._mask_opt, "mask_dirname", "masks_sam"))
        mask_root = os.path.join(self._source_path, self._images_subdir, mask_dirname)
        if not os.path.isdir(mask_root):
            return

        # Collect candidate object ids from existing mask files.
        obj_ids: List[int] = []
        obj_set = set()

        cams = self.getTrainCameras()
        if not cams:
            return

        max_objs = int(getattr(self._mask_opt, "mask_max_objects", 16))
        for cam in cams[: min(len(cams), 64)]:
            stem = os.path.splitext(os.path.basename(cam.image_name))[0]
            d = os.path.join(mask_root, stem)
            if not os.path.isdir(d):
                continue
            try:
                for fn in os.listdir(d):
                    if not (fn.startswith("obj_") and fn.endswith(".png")):
                        continue
                    try:
                        oid = int(fn[len("obj_") : len("obj_") + 4])
                    except Exception:
                        continue
                    if oid not in obj_set:
                        obj_set.add(oid)
                        obj_ids.append(oid)
            except FileNotFoundError:
                continue

        if not obj_ids:
            return

        # Use smallest ids to make the selection deterministic.
        obj_ids = sorted(obj_set)[:max_objs]

        # Choose voting views that actually contain each obj_id mask file.
        cam_by_stem: Dict[str, object] = {}
        for cam in cams:
            stem = os.path.splitext(os.path.basename(cam.image_name))[0]
            cam_by_stem[stem] = cam

        vote_views: List[object] = []
        used = set()
        for obj_id in obj_ids:
            found = None
            for stem, cam in cam_by_stem.items():
                p = os.path.join(mask_root, stem, f"obj_{int(obj_id):04d}.png")
                if os.path.exists(p):
                    found = cam
                    break
            if found is not None:
                key = os.path.splitext(os.path.basename(found.image_name))[0]
                if key not in used:
                    used.add(key)
                    vote_views.append(found)

        # Fallback: if none found (unexpected), sample a few views.
        if not vote_views:
            max_views = 24
            if len(cams) <= max_views:
                vote_views = cams
            else:
                step = max(int(len(cams) // max_views), 1)
                vote_views = cams[::step][:max_views]

        xyz = self.gaussians.get_xyz.detach()  # (N,3) on cuda
        N = int(xyz.shape[0])
        if N == 0:
            return

        votes = torch.zeros((len(obj_ids), N), device=xyz.device, dtype=torch.int16)

        def _load_mask(stem: str, obj_id: int, H: int, W: int, device) -> Optional[torch.Tensor]:
            p = os.path.join(mask_root, stem, f"obj_{int(obj_id):04d}.png")
            if not os.path.exists(p):
                return None
            m = Image.open(p).convert("L")
            arr = torch.from_numpy(np.array(m, dtype=np.uint8)).float() / 255.0
            if int(arr.shape[0]) != int(H) or int(arr.shape[1]) != int(W):
                arr = F.interpolate(arr[None, None, ...], size=(int(H), int(W)), mode="nearest")[0, 0]
            return arr[None, ...].to(device=device)

        # Vote for each object by checking whether projected gaussian centers fall inside masks.
        for cam in vote_views:
            H, W = int(cam.image_height), int(cam.image_width)
            stem = os.path.splitext(os.path.basename(cam.image_name))[0]
            # Project to NDC
            ndc = geom_transform_points(xyz, cam.full_proj_transform)
            x = ndc[:, 0]
            y = ndc[:, 1]
            z = ndc[:, 2]
            ix = ((x + 1.0) * 0.5 * float(W)).long()
            iy = ((1.0 - y) * 0.5 * float(H)).long()
            valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (z > 0)
            if not bool(valid.any().item()):
                continue
            vidx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            ixv = ix[vidx]
            iyv = iy[vidx]

            for oi, obj_id in enumerate(obj_ids):
                m = _load_mask(stem, obj_id, H, W, xyz.device)
                if m is None:
                    continue
                inside = (m[0, iyv, ixv] > 0.5).to(votes.dtype)
                votes[oi, vidx] += inside

        # Keep it permissive: export gaussians that hit each mask at least once.
        # This avoids empty exports when object IDs are not perfectly consistent across all views.
        min_votes = 1

        out_dir = os.path.join(point_cloud_path, "objects")
        os.makedirs(out_dir, exist_ok=True)
        for oi, obj_id in enumerate(obj_ids):
            sel = votes[oi] >= min_votes
            if int(sel.sum().item()) <= 0:
                continue
            out_ply = os.path.join(out_dir, f"obj_{int(obj_id):04d}.ply")
            self.gaussians.save_ply_masked(out_ply, sel)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
