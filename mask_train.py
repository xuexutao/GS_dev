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
    
    # Inpaint parameters 
    parser.add_argument("--do_inpaint", action="store_true", help="Run MVInpainter to fill background")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mask_dirname = "masks_sam"

    images_dir_for_resolution = f"images_{args.resolution}" if args.resolution > 1 else "images"
    expected_mask_path = os.path.join(args.source_path, images_dir_for_resolution, mask_dirname)

    # 1. SAM Mask Generation (Optional)
    if args.mask_object:
        # Check if masks already exist, skip if we don't want to overwrite
        if not os.path.exists(expected_mask_path):
            print(f"Masks not found at {expected_mask_path}. Generating masks for '{args.mask_object}'...")
            # Note: generate_multiview_sam_masks.py currently does Auto-masking.
            # If your script supports prompt-based SAM via Grounded-SAM, you'd call that here.
            # For now, we will call the existing auto-mask script which clusters objects.
            sam_cmd = f"""
            python generate_multiview_sam_masks.py \\
                --source_path {args.source_path} \\
                --images_subdir {images_dir_for_resolution} \\
                --out_subdir {images_dir_for_resolution}/{mask_dirname} \\
                --sam_checkpoint {args.sam_checkpoint} \\
                --sam_model_type vit_h
            """
            run_command(sam_cmd, f"Running SAM to generate masks in {images_dir_for_resolution}/{mask_dirname}")
        else:
            print(f"✅ Masks already exist at {expected_mask_path}, skipping SAM generation.")

    # 2. MVInpainter Background Completion (Optional)
    if args.do_inpaint:
        inpaint_cmd = f"""
        cd submodules/MVInpainter && python run_pipeline.py \\
            --data_dir ../../{args.source_path} \\
            --mask_dir {images_dir_for_resolution}/{mask_dirname} \\
            --output_dir ../../{args.output_dir}/inpainted_images
        """
        run_command(inpaint_cmd, "Inpainting Background")

    # 3. Train Foreground Object (Masked Rasterizer)
    if args.mask_object or os.path.exists(expected_mask_path):
        fg_output = os.path.join(args.output_dir, f"foreground_{args.mask_object_id}")
        cmd_fg = f"""
        python train.py -s {args.source_path} \\
            -r {args.resolution} \\
            -m {fg_output} \\
            --iterations {args.iterations} \\
            --mask_dirname {mask_dirname} \\
            --mask_only \\
            --mask_use_masked_rasterizer \\
            --mask_object_id {args.mask_object_id}
        """
        run_command(cmd_fg, f"Training Foreground Object (Masked Rasterizer, Object ID {args.mask_object_id})")

    # 4. Train Background Scene
    bg_output = os.path.join(args.output_dir, "background_scene")
    images_flag = f"--images {args.output_dir}/inpainted_images" if args.do_inpaint else ""
    
    cmd_bg = f"""
    python train.py -s {args.source_path} \\
        -r {args.resolution} \\
        -m {bg_output} \\
        --iterations {args.iterations} \\
        {images_flag}
    """
    run_command(cmd_bg, "Training Background Scene")

    print(f"\n✅ Pipeline finished successfully!")
    if args.mask_object or os.path.exists(expected_mask_path):
        print(f"Foreground Asset: {fg_output}/point_cloud/iteration_{args.iterations}/point_cloud.ply")
    print(f"Background Asset: {bg_output}/point_cloud/iteration_{args.iterations}/point_cloud.ply")

if __name__ == "__main__":
    main()
