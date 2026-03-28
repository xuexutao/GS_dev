#!/bin/bash
cd /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev
python test_assign.py \
  --source_path /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room \
  --model_path /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/output/model_output_0328 \
  --iteration 30000 \
  --semantic_labels /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room/images_4/masks_sam/labels.json \
  --images images_4 \
  --mask_dirname masks_sam \
  --max_objects 10 2>&1