import os
import sys
import subprocess
import argparse
import glob

def run_command(cmd, desc=""):
    print(f"\n{'='*80}")
    print(f"🚀 {desc}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    
    # Run command and pipe output to stdout
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    process.wait()
    if process.returncode != 0:
        print(f"❌ Error during: {desc}")
        sys.exit(process.returncode)

def main():
    parser = argparse.ArgumentParser(description="End-to-End Masked 3DGS Training Pipeline")
    parser.add_argument("--source_path", "-s", required=True, help="Path to COLMAP dataset")
    parser.add_argument("--output_dir", "-o", required=True, help="Base output directory")
    parser.add_argument("--resolution", "-r", type=int, default=8, help="Image downscale resolution")
    parser.add_argument("--iterations", "-i", type=int, default=1000, help="Number of training iterations")
    
    # SAM parameters
    parser.add_argument("--mask_object", type=str, default="", help="If specified, will run SAM with this prompt to generate masks")
    parser.add_argument("--sam_checkpoint", type=str, default="submodules/segment-anything/weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--mask_object_id", type=int, default=0, help="Object ID to extract (after clustering)")
    parser.add_argument("--sam_max_images", type=int, default=0, help="Debug: limit SAM mask generation to first N images (0 means all)")
    
    # Inpaint parameters 
    parser.add_argument("--do_inpaint", action="store_true", help="Run MVInpainter to fill background")
    parser.add_argument("--dual_w_fg", type=float, default=1.0, help="Foreground loss weight (dual training)")
    parser.add_argument("--dual_w_bg", type=float, default=1.0, help="Background loss weight (dual training)")
    parser.add_argument("--inpaint_max_images", type=int, default=-1, help="Debug: limit MVInpainter to first N images (-1 means all)")
    parser.add_argument(
        "--inpaint_python",
        type=str,
        default="python3.9",
        help="Python executable for MVInpainter (default python3.9; use to bypass diffusers/huggingface_hub version mismatch in gaussian env)",
    )
    
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    # Use the same Python as mask_train.py for training (keeps CUDA extensions ABI-consistent).
    py = sys.executable

    os.makedirs(args.output_dir, exist_ok=True)
    mask_dirname = "masks_sam"

    # NOTE:
    # This repo supports specifying the images subdir via `--images` (see arguments/ModelParams).
    # To keep masks and training aligned, always use the same images subdir for:
    # - generate_multiview_sam_masks.py (--images_subdir / --out_subdir)
    # - train.py (--images)
    images_subdir = f"images_{args.resolution}" if args.resolution > 1 else "images"
    # If the pre-downscaled folder doesn't exist, fall back to the default images folder
    # and rely on `-r/--resolution` to downscale during training.
    if images_subdir != "images" and not os.path.isdir(os.path.join(args.source_path, images_subdir)):
        print(f"⚠️  {images_subdir} not found under source_path; falling back to 'images'.")
        images_subdir = "images"

    expected_mask_path = os.path.join(args.source_path, images_subdir, mask_dirname)

    def _cmd(parts):
        # Build a safe single-line command (avoid newline/\\ continuation pitfalls).
        return " ".join(str(p) for p in parts if str(p).strip() != "")

    # 1. SAM Mask Generation (Auto)
    # 用户常见用法是只提供 --mask_object_id，但 masks_sam 可能还未生成；这里自动补齐。
    has_any_masks = False
    if os.path.isdir(expected_mask_path):
        if len(glob.glob(os.path.join(expected_mask_path, "*", "obj_*.png"))) > 0:
            has_any_masks = True

    if not has_any_masks:
        print(f"Masks not found at {expected_mask_path}. Auto-generating multi-view SAM masks...")
        sam_parts = [
            py,
            f"{repo_root}/generate_multiview_sam_masks.py",
            "--source_path",
            args.source_path,
            "--images_subdir",
            images_subdir,
            "--out_subdir",
            f"{images_subdir}/{mask_dirname}",
            "--sam_checkpoint",
            args.sam_checkpoint,
            "--sam_model_type",
            "vit_h",
        ]
        if int(getattr(args, "sam_max_images", 0)) > 0:
            sam_parts += ["--max_images", int(args.sam_max_images)]
        sam_cmd = _cmd(sam_parts)
        run_command(sam_cmd, f"Running SAM to generate masks in {images_subdir}/{mask_dirname}")
    else:
        print(f"✅ Masks already exist at {expected_mask_path}, skipping SAM generation.")

    # 2. MVInpainter Background Completion (Optional)
    if args.do_inpaint:
        image_dir = os.path.join(args.source_path, images_subdir)
        inpaint_mask_dir = os.path.join(args.output_dir, "mvinpainter_masks")
        inpaint_out_dir = os.path.join(args.output_dir, "inpainted_images")
        os.makedirs(inpaint_mask_dir, exist_ok=True)
        os.makedirs(inpaint_out_dir, exist_ok=True)

        # Flatten selected object mask to MVInpainter format: <mask_dir>/<image_stem>.png
        img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.JPG")))
        for p in img_paths:
            base = os.path.basename(p)
            stem, ext = os.path.splitext(base)
            src_mask = os.path.join(expected_mask_path, stem, f"obj_{int(args.mask_object_id):04d}.png")
            if not os.path.exists(src_mask):
                # write empty mask so MVInpainter copies image
                dst_mask = os.path.join(inpaint_mask_dir, base.replace(ext, ".png"))
                if not os.path.exists(dst_mask):
                    from PIL import Image
                    import numpy as np
                    img0 = Image.open(p)
                    empty = Image.fromarray(np.zeros((img0.size[1], img0.size[0]), dtype=np.uint8))
                    empty.save(dst_mask)
                continue
            dst_mask = os.path.join(inpaint_mask_dir, base.replace(ext, ".png"))
            if not os.path.exists(dst_mask):
                run_command(f"cp {src_mask} {dst_mask}", f"Prepare MVInpainter mask: {base}")

        # If outputs already exist (e.g., rerun), skip heavy inpainting.
        in_imgs = sorted(glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.JPG")))
        out_imgs = sorted(glob.glob(os.path.join(inpaint_out_dir, "*.png")) + glob.glob(os.path.join(inpaint_out_dir, "*.jpg")) + glob.glob(os.path.join(inpaint_out_dir, "*.JPG")))
        max_images = int(getattr(args, "inpaint_max_images", -1))
        expected = len(in_imgs) if max_images <= 0 else min(len(in_imgs), max_images)

        if len(out_imgs) >= expected and expected > 0:
            print(f"✅ Inpainted images already exist: {len(out_imgs)}/{expected}, skipping MVInpainter.")
        else:
            # MVInpainter has tighter dependency constraints; allow using another Python (default python3.9).
            inpaint_py = str(getattr(args, "inpaint_python", "python3.9") or "python3.9")
            inpaint_parts = [
                inpaint_py,
                f"{repo_root}/submodules/MVInpainter/run_pipeline.py",
                "--only_inpaint",
                "--image_dir",
                image_dir,
                "--mask_dir",
                inpaint_mask_dir,
                "--inpaint_out",
                inpaint_out_dir,
            ]
            if max_images > 0:
                inpaint_parts += ["--max_images", max_images]
            inpaint_cmd = _cmd(inpaint_parts)
            run_command(inpaint_cmd, "Running MVInpainter inpainting (background completion)")

    # 3. Train Dual Foreground/Background (single run)
    if not (args.mask_object or os.path.exists(expected_mask_path)):
        raise RuntimeError(f"Masks not found at {expected_mask_path}. Please run --mask_object first or provide masks on disk.")

    dual_output = os.path.join(args.output_dir, f"dual_fg_bg_obj{int(args.mask_object_id):04d}")
    bg_images_arg = os.path.join(args.output_dir, "inpainted_images") if args.do_inpaint else os.path.join(args.source_path, images_subdir)

    cmd_dual = _cmd([
        py,
        f"{repo_root}/train.py",
        "-s",
        args.source_path,
        "-r",
        int(args.resolution),
        "-m",
        dual_output,
        "--images",
        images_subdir,
        "--iterations",
        int(args.iterations),
        "--mask_dirname",
        mask_dirname,
        "--mask_object_id",
        int(args.mask_object_id),
        "--dual_enable",
        "--dual_bg_images",
        bg_images_arg,
        "--dual_w_fg",
        float(args.dual_w_fg),
        "--dual_w_bg",
        float(args.dual_w_bg),
        "--dual_use_masked_rasterizer",
        "--mask_use_masked_rasterizer",
    ])
    run_command(cmd_dual, "Training Dual Foreground/Background (masked FG + inpainted BG)")

    print(f"\n✅ Pipeline finished successfully!")
    print(f"Foreground Asset: {dual_output}/point_cloud/iteration_{args.iterations}/point_cloud.ply")
    print(f"Background Asset: {dual_output}/point_cloud/iteration_{args.iterations}/point_cloud_bg.ply")

if __name__ == "__main__":
    main()
