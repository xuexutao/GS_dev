#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    print("plyfile not installed. Trying to install...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData

ply_path = 'output/model_output_0328/point_cloud/iteration_30000/point_cloud.ply'
print(f"Reading PLY file: {ply_path}")
plydata = PlyData.read(ply_path)
vertices = plydata['vertex']
print(f"Number of vertices: {len(vertices)}")
print(f"Available properties: {list(vertices.properties)}")

if 'semantic' in vertices.properties:
    sem = vertices['semantic']
    print(f"Semantic shape: {sem.shape}")
    print(f"Semantic dtype: {sem.dtype}")
    unique_vals = np.unique(sem)
    print(f"Unique semantic values: {unique_vals}")
    print(f"Number of unique values: {len(unique_vals)}")
    # Count distribution
    for val in unique_vals:
        count = np.sum(sem == val)
        print(f"  Value {val}: {count} points")
    # Check if any values are not -1
    non_negative = sem[sem != -1]
    print(f"Points with semantic != -1: {len(non_negative)}")
    if len(non_negative) > 0:
        print(f"Sample non-negative values: {non_negative[:10]}")
else:
    print("No semantic property found")