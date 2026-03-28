#!/usr/bin/env python3
import os
import sys
import glob
from collections import defaultdict

mask_root = '/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room/images_4/masks_sam'
stem_dirs = [d for d in glob.glob(os.path.join(mask_root, "*")) if os.path.isdir(d)]
print(f"Found {len(stem_dirs)} view directories")
object_mask_map = defaultdict(list)

for stem_dir in stem_dirs[:5]:  # only first 5
    stem = os.path.basename(stem_dir)
    mask_files = glob.glob(os.path.join(stem_dir, "obj_*.png"))
    print(f"  {stem}: {len(mask_files)} mask files")
    for mf in mask_files[:2]:  # first 2
        fname = os.path.basename(mf)
        try:
            obj_id = int(fname.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        print(f"    obj_{obj_id}: {mf}")
        object_mask_map[obj_id].append((stem, mf))

print(f"\nTotal unique object IDs: {len(object_mask_map)}")
for obj_id, items in list(object_mask_map.items())[:10]:
    print(f"  obj_{obj_id}: {len(items)} views")