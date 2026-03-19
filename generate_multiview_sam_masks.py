import argparse
import os
import glob

import numpy as np
from PIL import Image

from utils.multiview_sam_mask import (
    SamAutoMaskParams,
    generate_multiview_consistent_masks,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SAM masks and enforce multi-view consistency using COLMAP tracks."
    )
    parser.add_argument("--source_path", required=True, help="Dataset root, contains images/ and sparse/0")
    parser.add_argument("--images_subdir", default="images")
    parser.add_argument("--sparse_subdir", default=os.path.join("sparse", "0"))
    parser.add_argument("--out_subdir", default=os.path.join("images", "masks_sam"))
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true", help="Do not load SAM; write a single full-image mask per view for sanity check")
    parser.add_argument("--min_shared_points", type=int, default=20)
    parser.add_argument("--max_points_per_image", type=int, default=5000)
    parser.add_argument("--max_images", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--summary_topk",
        type=int,
        default=10,
        help="After writing masks, print top-K objects by total mask area across views (0 disables)",
    )

    # SAM AutomaticMaskGenerator params
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.86)
    parser.add_argument("--stability_score_thresh", type=float, default=0.92)
    parser.add_argument("--min_mask_region_area", type=int, default=256)

    args = parser.parse_args()
    sam_params = SamAutoMaskParams(
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )

    out_dir = generate_multiview_consistent_masks(
        source_path=args.source_path,
        images_subdir=args.images_subdir,
        sparse_subdir=args.sparse_subdir,
        out_subdir=args.out_subdir,
        model_type=args.sam_model_type,
        checkpoint=args.sam_checkpoint,
        device=args.device,
        sam_params=sam_params,
        min_shared_points=args.min_shared_points,
        max_points_per_image=(args.max_points_per_image if args.max_points_per_image > 0 else None),
        max_images=(args.max_images if args.max_images and args.max_images > 0 else None),
        dry_run=args.dry_run,
    )

    print(f"Masks written to: {out_dir}")

    if int(args.summary_topk) > 0:
        # Summarize object ids by total area to help choose `--mask_object_id` later.
        obj_area = {}
        obj_views = {}
        for stem_dir in sorted(glob.glob(os.path.join(out_dir, "*"))):
            if not os.path.isdir(stem_dir):
                continue
            stem = os.path.basename(stem_dir)
            for p in glob.glob(os.path.join(stem_dir, "obj_*.png")):
                name = os.path.basename(p)
                try:
                    obj_id = int(name.split("_")[1].split(".")[0])
                except Exception:
                    continue
                m = np.array(Image.open(p).convert("L"), dtype=np.uint8)
                area = int((m > 127).sum())
                obj_area[obj_id] = obj_area.get(obj_id, 0) + area
                obj_views.setdefault(obj_id, set()).add(stem)

        ranked = sorted(obj_area.items(), key=lambda kv: -kv[1])
        topk = ranked[: int(args.summary_topk)]
        if topk:
            print("\n[Mask Summary] top objects by total area:")
            for obj_id, area in topk:
                print(f"  obj_{obj_id:04d}: area={area} views={len(obj_views.get(obj_id, []))}")


if __name__ == "__main__":
    main()
