#!/usr/bin/env python3
import sys
import numpy as np

def read_ply_header(ply_path):
    """Read PLY file header to see properties"""
    with open(ply_path, 'rb') as f:
        lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore')
            lines.append(line)
            if 'end_header' in line:
                break
        return lines

if __name__ == "__main__":
    ply_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/output/model_output_0328/point_cloud_with_semantic_test.ply"
    print(f"Reading PLY header from {ply_path}")
    header = read_ply_header(ply_path)
    for line in header:
        print(line.strip())