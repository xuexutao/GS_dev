#!/bin/bash
cd /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev
python assign_semantic_to_trained_model.py \
  --source_path /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room \
  --model_path /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/output/model_output_0328 \
  --iteration 30000 \
  --semantic_labels /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room/images_4/masks_sam/labels.json \
  --images images_4 \
  --mask_dirname masks_sam \
  --mask_only \
  --mask_object_id -1 2>&1