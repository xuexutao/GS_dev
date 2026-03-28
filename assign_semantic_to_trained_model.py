#!/usr/bin/env python3
"""
为已训练的3DGS模型分配语义标签（后处理）
使用已有的 masks 和 labels.json 文件，将语义标签分配给高斯点云
"""

import os
import sys
import json
import argparse
from argparse import ArgumentParser, Namespace
import torch
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams


def main():
    parser = argparse.ArgumentParser(description="Assign semantic labels to trained 3DGS model")
    parser.add_argument("--source_path", required=True, help="Dataset root (contains images/ and sparse/0)")
    parser.add_argument("--model_path", required=True, help="Path to trained model output directory")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration number to load")
    parser.add_argument("--semantic_labels", required=True, help="Path to labels.json file")
    parser.add_argument("--images", default="images", help="Images subdirectory")
    parser.add_argument("--resolution", type=int, default=4, help="Resolution scale")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--mask_dirname", default="masks_sam", help="Mask subdirectory under images")
    parser.add_argument("--mask_only", action="store_true", help="Mask only mode")
    parser.add_argument("--mask_object_id", type=int, default=-1, help="Specific object ID to mask")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # 加载语义标签文件
    print(f"Loading semantic labels from {args.semantic_labels}")
    with open(args.semantic_labels, 'r') as f:
        semantic_labels = json.load(f)
    print(f"Loaded {len(semantic_labels)} semantic labels")
    
    # 创建 ModelParams 对象
    # 创建 ModelParams 对象（使用与 train.py 相同的模式）
    dummy_parser = ArgumentParser()
    lp = ModelParams(dummy_parser)
    op = OptimizationParams(dummy_parser)
    pp = PipelineParams(dummy_parser)
    # 创建命名空间包含我们的参数
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
    # 提取参数组
    model_params = lp.extract(ns)
    opt_params = op.extract(ns)
    pipe_params = pp.extract(ns)
    
    # 创建高斯模型
    gaussians = GaussianModel(args.sh_degree, "adam")  # 优化器类型不影响
    
    # 加载训练好的点云
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)
    
    print(f"Loading Gaussians from {ply_path}")
    gaussians.load_ply(ply_path, use_train_test_exp=False, device=args.device)
    
    # 创建 mask_opt 命名空间
    mask_opt = Namespace(
        mask_dirname=args.mask_dirname,
        mask_only=args.mask_only,
        mask_object_id=args.mask_object_id
    )
    # 如果提供了 mask_dirname，构建完整的 mask_dir 路径
    if args.mask_dirname:
        mask_dir = os.path.join(args.source_path, args.images, args.mask_dirname)
        if os.path.exists(mask_dir):
            mask_opt.mask_dir = mask_dir
            print(f"Set mask_opt.mask_dir to: {mask_dir}")
    
    # 创建场景对象（用于分配语义标签）
    scene = Scene(model_params, gaussians, load_iteration=args.iteration, mask_opt=mask_opt)
    
    # 分配语义标签
    scene.assign_semantic_labels(semantic_labels)
    
    # 保存带有语义标签的点云
    output_ply_path = os.path.join(args.model_path, f"point_cloud_with_semantic_iteration_{args.iteration}.ply")
    gaussians.save_ply(output_ply_path)
    print(f"Saved Gaussians with semantic labels to {output_ply_path}")
    
    # 同时保存到原目录（覆盖可选）
    gaussians.save_ply(ply_path)
    print(f"Overwritten original PLY file with semantic labels")
    
    # 打印统计信息
    assigned = (gaussians._semantic != -1).sum().item()
    total = gaussians._semantic.shape[0]
    print(f"\nSummary:")
    print(f"  Total Gaussians: {total}")
    print(f"  Labeled Gaussians: {assigned} ({assigned/total*100:.1f}%)")
    
    # 输出类别统计
    unique_labels = torch.unique(gaussians._semantic)
    for label in unique_labels:
        if label == -1:
            continue
        count = (gaussians._semantic == label).sum().item()
        print(f"  Label {label}: {count} points")


if __name__ == "__main__":
    main()