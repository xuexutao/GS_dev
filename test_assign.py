#!/usr/bin/env python3
import os
import sys
import json
import argparse
from argparse import ArgumentParser, Namespace
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

def main():
    parser = argparse.ArgumentParser(description="Test assign semantic labels to trained 3DGS model")
    parser.add_argument("--source_path", required=True, help="Dataset root")
    parser.add_argument("--model_path", required=True, help="Path to trained model output directory")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration number to load")
    parser.add_argument("--semantic_labels", required=True, help="Path to labels.json file")
    parser.add_argument("--images", default="images", help="Images subdirectory")
    parser.add_argument("--mask_dirname", default="masks_sam", help="Mask subdirectory under images")
    parser.add_argument("--max_objects", type=int, default=10, help="Maximum number of objects to process")
    
    args = parser.parse_args()
    
    # Load semantic labels file
    print(f"Loading semantic labels from {args.semantic_labels}")
    with open(args.semantic_labels, 'r') as f:
        semantic_labels = json.load(f)
    print(f"Loaded {len(semantic_labels)} semantic labels")
    
    # Limit to first max_objects
    if args.max_objects > 0:
        limited = {}
        for i, (k, v) in enumerate(semantic_labels.items()):
            if i >= args.max_objects:
                break
            limited[k] = v
        semantic_labels = limited
        print(f"Limited to {len(semantic_labels)} objects for testing")
    
    # Create ModelParams object
    dummy_parser = ArgumentParser()
    lp = ModelParams(dummy_parser)
    op = OptimizationParams(dummy_parser)
    pp = PipelineParams(dummy_parser)
    ns = Namespace(
        source_path=args.source_path,
        model_path=args.model_path,
        images=args.images,
        resolution=4,
        white_background=False,
        sh_degree=3,
        train_test_exp=False,
        data_device="cuda",
        eval=False,
        depths="",
        start_type="train"
    )
    model_params = lp.extract(ns)
    opt_params = op.extract(ns)
    pipe_params = pp.extract(ns)
    
    # Create Gaussian model
    gaussians = GaussianModel(3, "adam")
    
    # Load trained point cloud
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)
    
    print(f"Loading Gaussians from {ply_path}")
    gaussians.load_ply(ply_path)
    
    # Create mask_opt namespace
    mask_opt = Namespace(
        mask_dirname=args.mask_dirname,
        mask_only=False,
        mask_object_id=-1
    )
    
    # Create scene object
    scene = Scene(model_params, gaussians, mask_opt=mask_opt)
    
    # Assign semantic labels
    scene.assign_semantic_labels(semantic_labels)
    
    # Save with semantic labels
    output_ply_path = os.path.join(args.model_path, f"point_cloud_with_semantic_test.ply")
    gaussians.save_ply(output_ply_path)
    print(f"Saved Gaussians with semantic labels to {output_ply_path}")
    
    # Print statistics
    assigned = (gaussians._semantic != -1).sum().item()
    total = gaussians._semantic.shape[0]
    print(f"\nSummary:")
    print(f"  Total Gaussians: {total}")
    print(f"  Labeled Gaussians: {assigned} ({assigned/total*100:.1f}%)")
    
    unique_labels = torch.unique(gaussians._semantic)
    for label in unique_labels:
        if label == -1:
            continue
        count = (gaussians._semantic == label).sum().item()
        print(f"  Label {label}: {count} points")

if __name__ == "__main__":
    main()