#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argparse import Namespace

# Simulate mask_opt
mask_opt = Namespace(mask_dirname='masks_sam')
source_path = 'data/gs_data/room'
images_subdir = 'images_4'  # correct
# images_subdir = 'images'   # wrong

mask_root = os.path.join(source_path, images_subdir, mask_opt.mask_dirname)
print(f"Constructed mask directory: {mask_root}")
print(f"Exists: {os.path.isdir(mask_root)}")

# Also test with mask_dir attribute
mask_opt.mask_dir = os.path.join(source_path, images_subdir, mask_opt.mask_dirname)
print(f"mask_opt.mask_dir: {mask_opt.mask_dir}")
print(f"Exists: {os.path.isdir(mask_opt.mask_dir)}")