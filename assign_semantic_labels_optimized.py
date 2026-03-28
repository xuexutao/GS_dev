#!/usr/bin/env python3
"""
优化版的语义标签分配脚本
使用 per-camera 处理，大幅提升性能
"""

import os
import sys
import json
import argparse
from argparse import ArgumentParser, Namespace
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import geom_transform_points


def assign_semantic_labels_optimized(
    gaussians, 
    scene, 
    semantic_labels, 
    mask_root,
    category_to_id=None,
    max_views_per_object=5
):
    """优化版的语义标签分配函数
    
    参数:
        gaussians: GaussianModel 实例
        scene: Scene 实例（用于获取相机）
        semantic_labels: 从对象ID到类别信息的映射
        mask_root: 掩码根目录
        category_to_id: 类别到ID的映射，如果为None则自动创建
        max_views_per_object: 每个对象最多使用的视图数（用于投票）
    """
    if semantic_labels is None or len(semantic_labels) == 0:
        print("Warning: No semantic labels provided")
        return
    
    print(f"Assigning semantic labels to {len(semantic_labels)} objects (optimized)...")
    
    # 创建类别到ID的映射
    if category_to_id is None:
        categories = set()
        for obj_info in semantic_labels.values():
            categories.add(obj_info["category"])
        categories = sorted(list(categories))
        category_to_id = {cat: i for i, cat in enumerate(categories)}
        print(f"Created category mapping: {category_to_id}")
    
    # 获取所有对象ID
    obj_ids = []
    for obj_id_str in semantic_labels.keys():
        try:
            obj_id = int(obj_id_str)
            obj_ids.append(obj_id)
        except ValueError:
            print(f"Warning: Invalid object ID format: {obj_id_str}")
            continue
    
    if not obj_ids:
        print("Warning: No valid object IDs found")
        return
    
    # 获取训练相机
    cams = scene.getTrainCameras()
    if not cams:
        print("Warning: No training cameras available")
        return
    
    # 按图像词干组织相机
    cam_by_stem = {}
    for cam in cams:
        stem = os.path.splitext(os.path.basename(cam.image_name))[0]
        cam_by_stem[stem] = cam
    
    # 预计算每个对象在哪些视图中存在掩码
    print("Precomputing mask existence...")
    obj_views = {}  # obj_id -> list of (stem, cam)
    for obj_id in tqdm(obj_ids, desc="Scanning masks"):
        views = []
        for stem, cam in cam_by_stem.items():
            mask_path = os.path.join(mask_root, stem, f"obj_{obj_id:04d}.png")
            if os.path.exists(mask_path):
                views.append((stem, cam))
        if views:
            # 限制每个对象使用的视图数
            if len(views) > max_views_per_object:
                # 选择前 max_views_per_object 个视图
                views = views[:max_views_per_object]
            obj_views[obj_id] = views
    
    print(f"Found masks for {len(obj_views)} objects")
    
    # 按视图组织对象，以便批量处理
    view_objects = {}  # stem -> list of (obj_id, category_id)
    for obj_id, views in obj_views.items():
        obj_info = semantic_labels.get(str(obj_id))
        if obj_info is None:
            continue
        category = obj_info["category"]
        category_id = category_to_id.get(category)
        if category_id is None:
            continue
        
        for stem, cam in views:
            if stem not in view_objects:
                view_objects[stem] = []
            view_objects[stem].append((obj_id, category_id, cam))
    
    print(f"Processing {len(view_objects)} unique views...")
    
    xyz = gaussians.get_xyz.detach()  # (N,3)
    N = int(xyz.shape[0])
    if N == 0:
        print("Warning: No Gaussians to assign labels to")
        return
    
    # 初始化语义标签为-1（未标记）
    semantic_tensor = torch.full((N,), -1, dtype=torch.long, device=xyz.device)
    
    # 处理每个视图
    for stem, objects in tqdm(view_objects.items(), desc="Processing views"):
        if not objects:
            continue
        
        # 获取第一个对象的相机（同一个stem的所有对象共享同一个相机）
        _, _, cam = objects[0]
        H, W = int(cam.image_height), int(cam.image_width)
        
        # 投影高斯点到当前视图
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
        
        # 为当前视图预加载所有对象的掩码
        mask_cache = {}
        for obj_id, category_id, _ in objects:
            mask_path = os.path.join(mask_root, stem, f"obj_{obj_id:04d}.png")
            if not os.path.exists(mask_path):
                continue
            
            # 加载掩码
            m = Image.open(mask_path).convert("L")
            arr = torch.from_numpy(np.array(m, dtype=np.uint8)).float() / 255.0
            if int(arr.shape[0]) != int(H) or int(arr.shape[1]) != int(W):
                arr = F.interpolate(arr[None, None, ...], size=(int(H), int(W)), mode="nearest")[0, 0]
            mask_cache[obj_id] = arr[None, ...].to(device=xyz.device)
        
        # 为每个对象分配标签
        for obj_id, category_id, _ in objects:
            if obj_id not in mask_cache:
                continue
            
            m = mask_cache[obj_id]
            # 检查哪些高斯点在掩码内
            inside = m[0, iyv, ixv] > 0.5
            if inside.any():
                # 为在掩码内的高斯点分配类别ID
                mask_indices = vidx[inside]
                for idx in mask_indices:
                    if semantic_tensor[idx] == -1:  # 只分配未标记的点
                        semantic_tensor[idx] = category_id
    
    # 将语义标签分配给高斯模型
    gaussians._semantic = semantic_tensor
    
    # 统计分配情况
    assigned = (semantic_tensor != -1).sum().item()
    print(f"Assigned semantic labels to {assigned}/{N} Gaussians ({assigned/N*100:.1f}%)")
    
    # 打印每个类别的统计信息
    category_counts = {}
    for cat_name, cat_id in category_to_id.items():
        count = (semantic_tensor == cat_id).sum().item()
        if count > 0:
            category_counts[cat_name] = count
    
    print("Category distribution:")
    for cat_name, count in category_counts.items():
        print(f"  {cat_name}: {count} Gaussians")
    
    return category_to_id


def main():
    parser = argparse.ArgumentParser(description="Optimized semantic label assignment for trained 3DGS model")
    parser.add_argument("--source_path", required=True, help="Dataset root (contains images/ and sparse/0)")
    parser.add_argument("--model_path", required=True, help="Path to trained model output directory")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration number to load")
    parser.add_argument("--semantic_labels", required=True, help="Path to labels.json file")
    parser.add_argument("--images", default="images", help="Images subdirectory")
    parser.add_argument("--resolution", type=int, default=4, help="Resolution scale")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--mask_dirname", default="masks_sam", help="Mask subdirectory under images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--max_views_per_object", type=int, default=3, help="Maximum number of views to use per object")
    
    args = parser.parse_args()
    
    # 加载语义标签文件
    print(f"Loading semantic labels from {args.semantic_labels}")
    with open(args.semantic_labels, 'r') as f:
        semantic_labels = json.load(f)
    print(f"Loaded {len(semantic_labels)} semantic labels")
    
    # 创建 ModelParams 对象
    dummy_parser = ArgumentParser()
    lp = ModelParams(dummy_parser)
    op = OptimizationParams(dummy_parser)
    pp = PipelineParams(dummy_parser)
    
    ns = Namespace(
        source_path=args.source_path,
        model_path=args.model_path,
        images=args.images,
        resolution=args.resolution,
        white_background=args.white_background,
        sh_degree=args.sh_degree,
        train_test_exp=False,
        data_device=args.device,
        eval=False,
        depths="",
        start_type="train"
    )
    
    model_params = lp.extract(ns)
    opt_params = op.extract(ns)
    pipe_params = pp.extract(ns)
    
    # 创建高斯模型
    gaussians = GaussianModel(args.sh_degree, "adam")
    
    # 加载训练好的点云
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)
    
    print(f"Loading Gaussians from {ply_path}")
    gaussians.load_ply(ply_path, device=args.device)
    
    # 创建 mask_opt 命名空间
    mask_opt = Namespace(mask_dirname=args.mask_dirname)
    
    # 创建场景对象
    scene = Scene(model_params, gaussians, load_iteration=args.iteration, mask_opt=mask_opt)
    
    # 构建掩码根路径
    mask_root = os.path.join(args.source_path, args.images, args.mask_dirname)
    
    # 使用优化版函数分配语义标签
    category_to_id = assign_semantic_labels_optimized(
        gaussians=gaussians,
        scene=scene,
        semantic_labels=semantic_labels,
        mask_root=mask_root,
        category_to_id=None,
        max_views_per_object=args.max_views_per_object
    )
    
    # 保存带有语义标签的点云
    output_ply_path = os.path.join(args.model_path, f"point_cloud_with_semantic_optimized_iteration_{args.iteration}.ply")
    gaussians.save_ply(output_ply_path)
    print(f"Saved Gaussians with semantic labels to {output_ply_path}")
    
    # 同时保存类别映射
    if category_to_id:
        mapping_path = os.path.join(args.model_path, f"category_mapping_iteration_{args.iteration}.json")
        with open(mapping_path, 'w') as f:
            json.dump(category_to_id, f, indent=2)
        print(f"Saved category mapping to {mapping_path}")


if __name__ == "__main__":
    main()