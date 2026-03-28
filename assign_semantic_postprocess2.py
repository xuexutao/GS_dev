#!/usr/bin/env python3
"""
为已训练的3DGS模型分配语义标签（后处理）
使用已有的 masks 和 labels.json 文件，将语义标签分配给高斯点云
版本2：正确使用参数组，避免冲突
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams


def main():
    # 创建主解析器
    parser = argparse.ArgumentParser(description="Assign semantic labels to trained 3DGS model")
    
    # 添加自定义参数
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration number to load")
    parser.add_argument("--semantic_labels", required=True, help="Path to labels.json file")
    
    # 添加标准参数组
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # 解析参数
    args = parser.parse_args()
    
    # 提取参数组
    model_params = lp.extract(args)
    opt_params = op.extract(args)
    pipe_params = pp.extract(args)
    
    # 加载语义标签文件
    print(f"Loading semantic labels from {args.semantic_labels}")
    with open(args.semantic_labels, 'r') as f:
        semantic_labels = json.load(f)
    print(f"Loaded {len(semantic_labels)} semantic labels")
    
    # 创建高斯模型
    gaussians = GaussianModel(model_params.sh_degree, opt_params.optimizer_type)
    
    # 加载训练好的点云
    ply_path = os.path.join(model_params.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)
    
    print(f"Loading Gaussians from {ply_path}")
    gaussians.load_ply(ply_path)
    
    # 创建场景对象（用于分配语义标签）
    # 传递 load_iteration 以避免从 COLMAP 重新创建点云
    scene = Scene(model_params, gaussians, load_iteration=args.iteration, mask_opt=opt_params)
    
    # 分配语义标签
    scene.assign_semantic_labels(semantic_labels)
    
    # 保存带有语义标签的点云
    output_ply_path = os.path.join(model_params.model_path, f"point_cloud_with_semantic_iteration_{args.iteration}.ply")
    gaussians.save_ply(output_ply_path)
    print(f"Saved Gaussians with semantic labels to {output_ply_path}")
    
    # 同时保存到原目录（覆盖可选）
    # gaussians.save_ply(ply_path)
    # print(f"Overwritten original PLY file with semantic labels")
    
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