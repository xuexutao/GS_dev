#!/usr/bin/env python3
import sys
import numpy as np
import struct

def read_ply_semantic_sample(ply_path, sample_size=1000):
    """Read a sample of semantic values from binary PLY file"""
    # First read header to find semantic property index and vertex count
    with open(ply_path, 'rb') as f:
        header_lines = []
        vertex_count = 0
        properties = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_name, prop_type))
            elif line == 'end_header':
                break
        
        # Find semantic property index
        semantic_idx = -1
        for i, (name, _) in enumerate(properties):
            if name == 'semantic':
                semantic_idx = i
                break
        
        if semantic_idx == -1:
            print("Semantic property not found in PLY file")
            return
        
        # Calculate byte offset to semantic property
        # Need to know size of each property type
        type_sizes = {'float': 4, 'double': 8, 'int': 4, 'uint': 4, 'char': 1}
        offset = 0
        for i, (name, typ) in enumerate(properties):
            if i == semantic_idx:
                break
            offset += type_sizes.get(typ, 4)  # default 4 bytes
        
        # Total size per vertex
        vertex_size = sum(type_sizes.get(typ, 4) for _, typ in properties)
        
        # Read sample of semantic values
        f.seek(len(b'\n'.join([line.encode() for line in header_lines])) + 1)  # skip header
        
        # Read sample vertices
        sample_indices = np.random.choice(vertex_count, size=min(sample_size, vertex_count), replace=False)
        sample_indices.sort()
        
        semantic_values = []
        for idx in sample_indices:
            f.seek(len(b'\n'.join([line.encode() for line in header_lines])) + 1 + idx * vertex_size + offset)
            data = f.read(4)  # float is 4 bytes
            if len(data) == 4:
                val = struct.unpack('f', data)[0]
                semantic_values.append(val)
        
        return semantic_values, vertex_count

if __name__ == "__main__":
    ply_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/output/model_output_0328/point_cloud/iteration_30000/point_cloud.ply"
    print(f"Checking semantic values in {ply_path}")
    values, total = read_ply_semantic_sample(ply_path, sample_size=100)
    if values:
        print(f"Total vertices: {total}")
        print(f"Sampled {len(values)} semantic values:")
        unique = np.unique(values)
        print(f"Unique values: {unique}")
        print(f"Value range: [{min(values)}, {max(values)}]")
        
        # Count how many are -1 vs other values
        neg_one_count = sum(1 for v in values if abs(v + 1) < 1e-6)
        other_count = len(values) - neg_one_count
        print(f"Values == -1: {neg_one_count} ({neg_one_count/len(values)*100:.1f}%)")
        print(f"Values != -1: {other_count}")
        
        if other_count > 0:
            other_values = [v for v in values if abs(v + 1) >= 1e-6]
            print(f"Non-negative-one values: {other_values[:10]}")