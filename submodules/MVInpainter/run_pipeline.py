import sys
import os
import glob
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Add SAM to path
sys.path.append("/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/submodules/segment-anything")
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    print(f"Failed to import SAM: {e}")

# Transformers for OwlViT
from transformers import pipeline

# For 3D filtering
from read_write_model import read_model, write_model, Point3D

# Diffusers for Inpainting
from diffusers import StableDiffusionInpaintPipeline
from diffusers.models.attention_processor import AttentionProcessor

# -------------------------------------------------------------
# 1. SAM 2D Mask Generation
# -------------------------------------------------------------
def get_masks_for_bicycle(image_dir, mask_out_dir, device="cuda"):
    print("Loading OwlViT and SAM...")
    detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection", device=device)
    
    sam_checkpoint = "/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/submodules/segment-anything/weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.JPG")) + glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Just take first 10 for testing to avoid taking forever, or all if needed?
    # Prompt asks "把多视角图片里的自行车被扣掉的图都补全", but for testing, let's process first 10, or all if they are fast.
    # Let's do all.
    for img_path in tqdm(image_paths, desc="Generating Masks"):
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_out_dir, img_name.replace(".JPG", ".png").replace(".jpg", ".png"))
        if os.path.exists(mask_path):
            continue
            
        image = Image.open(img_path).convert("RGB")
        
        # 1. Detect bicycle
        predictions = detector(image, candidate_labels=["bicycle"], threshold=0.1)
        
        # If found, take the one with highest score
        if predictions:
            best_pred = max(predictions, key=lambda x: x["score"])
            box = best_pred["box"]
            input_box = np.array([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
            
            # 2. Segment with SAM
            image_np = np.array(image)
            predictor.set_image(image_np)
            
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            mask = masks[0]
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(mask_path)
        else:
            # Empty mask
            mask_img = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))
            mask_img.save(mask_path)

# -------------------------------------------------------------
# 2. COLMAP Point Cloud Filtering
# -------------------------------------------------------------
def filter_bicycle_from_sfm(sparse_in_dir, mask_dir, sparse_out_dir):
    print("Reading COLMAP model...")
    cameras, images, points3D = read_model(path=sparse_in_dir, ext=".bin")
    
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    masks = {}
    for mf in mask_files:
        masks[os.path.basename(mf)] = cv2.imread(mf, cv2.IMREAD_GRAYSCALE) > 127
        
    print(f"Loaded {len(masks)} masks. Filtering points...")
    
    # We will remove points that fall into the bicycle mask in majority of their observing views
    points_to_remove = set()
    
    for pt_id, point in tqdm(points3D.items(), desc="Filtering 3D Points"):
        hit_count = 0
        total_views = 0
        
        for img_id, pt2d_idx in zip(point.image_ids, point.point2D_idxs):
            img = images[img_id]
            cam = cameras[img.camera_id]
            img_name = img.name
            mask_name = img_name.replace(".JPG", ".png").replace(".jpg", ".png")
            
            if mask_name in masks:
                total_views += 1
                # Project 3D point
                xyz = point.xyz
                # R * xyz + T
                xyz_cam = img.qvec2rotmat() @ xyz + img.tvec
                
                # We can also just use the 2D observation directly! COLMAP stores it!
                # Wait, img.xys[pt2d_idx] is exactly the 2D coordinate of this 3D point in the image!
                xy = img.xys[pt2d_idx]
                x, y = int(xy[0]), int(xy[1])
                
                mask = masks[mask_name]
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x]:
                        hit_count += 1
                        
        # If the point is in the bicycle mask for more than 50% of its views
        if total_views > 0 and hit_count / total_views >= 0.5:
            points_to_remove.add(pt_id)
            
    print(f"Removing {len(points_to_remove)} points out of {len(points3D)}.")
    
    # Create new points3D dict
    new_points3D = {}
    for pt_id, point in points3D.items():
        if pt_id not in points_to_remove:
            new_points3D[pt_id] = point
            
    # Save the filtered model
    os.makedirs(sparse_out_dir, exist_ok=True)
    write_model(cameras, images, new_points3D, path=sparse_out_dir, ext=".bin")
    # Also save as PLY for visualization
    write_model(cameras, images, new_points3D, path=sparse_out_dir, ext=".ply")

# -------------------------------------------------------------
# 3. Multi-View Inpainting with Stable Diffusion
# -------------------------------------------------------------
class SharedAttentionProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size = hidden_states.shape[0]
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            # Self-attention: share K and V across all views
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            
            # Reshape to [1, batch_size * sequence_length, dim] and expand
            # Note: diffusers batch includes unconditional + conditional (2 * num_images)
            # We want each image to attend to all images in the same condition group
            chunk_size = batch_size // 2
            
            # Split uncoditional and conditional
            k_uncond, k_cond = key.chunk(2)
            v_uncond, v_cond = value.chunk(2)
            
            # Concat sequence dimension
            k_uncond_shared = k_uncond.reshape(1, -1, key.shape[-1]).expand(chunk_size, -1, -1)
            v_uncond_shared = v_uncond.reshape(1, -1, value.shape[-1]).expand(chunk_size, -1, -1)
            
            k_cond_shared = k_cond.reshape(1, -1, key.shape[-1]).expand(chunk_size, -1, -1)
            v_cond_shared = v_cond.reshape(1, -1, value.shape[-1]).expand(chunk_size, -1, -1)
            
            key = torch.cat([k_uncond_shared, k_cond_shared], dim=0)
            value = torch.cat([v_uncond_shared, v_cond_shared], dim=0)
        else:
            # Cross-attention
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        
        return hidden_states

def set_shared_attention(unet):
    for name, module in unet.named_modules():
        if "attn1" in name and hasattr(module, 'processor'):
            module.set_processor(SharedAttentionProcessor())

def inpaint_multiview(
    image_dir,
    mask_dir,
    out_dir,
    device="cuda",
    batch_size: int = 4,
    max_images: int = -1,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    prompt: str = "background, natural scene, photorealistic, high quality",
    negative_prompt: str = "artifacts, blur",
):
    print("Loading Stable Diffusion Inpainting pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to(device)
    
    # Apply multi-view shared attention
    set_shared_attention(pipe.unet)
    
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.JPG"))
        + glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    if max_images is not None and int(max_images) > 0:
        image_paths = image_paths[: int(max_images)]
    
    # Load all testing images and masks
    all_imgs = []
    all_masks = []
    valid_paths = []
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, img_name.replace(".JPG", ".png").replace(".jpg", ".png"))
        
        if not os.path.exists(mask_path):
            continue
            
        img_full = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_full.size
        # Run SD inpainting at 512 for stability, then resize back.
        img = img_full.resize((512, 512))
        mask = Image.open(mask_path).convert("L").resize((512, 512))
        
        # Only inpaint if there is a mask
        if np.array(mask).sum() > 0:
            all_imgs.append(img)
            all_masks.append(mask)
            valid_paths.append(img_path)
        else:
            # Just copy original
            img_full.save(os.path.join(out_dir, img_name))
            
    # Process
    if len(all_imgs) == 0:
        print("No masks found to inpaint.")
        return
        
    print(f"Inpainting {len(all_imgs)} images in chunks...")
    
    for i in tqdm(range(0, len(all_imgs), batch_size), desc="Inpainting chunks"):
        chunk_imgs = all_imgs[i:i+batch_size]
        chunk_masks = all_masks[i:i+batch_size]
        chunk_paths = valid_paths[i:i+batch_size]
        
        # Generate
        results = pipe(
            prompt=[prompt] * len(chunk_imgs),
            negative_prompt=[negative_prompt] * len(chunk_imgs),
            image=chunk_imgs,
            mask_image=chunk_masks,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale)
        ).images
        
        for res, p in zip(results, chunk_paths):
            # Resize back to original resolution
            full = Image.open(p).convert("RGB")
            ow, oh = full.size
            res = res.resize((ow, oh))
            res.save(os.path.join(out_dir, os.path.basename(p)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/bicycle/images_4")
    parser.add_argument("--mask_dir", type=str, default="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/submodules/MVInpainter/output/masks")
    parser.add_argument("--sparse_in", type=str, default="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/bicycle/sparse/0")
    parser.add_argument("--sparse_out", type=str, default="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/submodules/MVInpainter/output/sparse/0")
    parser.add_argument("--inpaint_out", type=str, default="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/submodules/MVInpainter/output/inpainted_images")
    parser.add_argument("--only_inpaint", action="store_true", help="Skip SAM/Colmap filtering, only run multi-view inpainting with provided masks")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--prompt", type=str, default="background, natural scene, photorealistic, high quality")
    parser.add_argument("--negative_prompt", type=str, default="artifacts, blur")
    
    args = parser.parse_args()
    
    if not args.only_inpaint:
        # 1. SAM segmentation
        print("=== Step 1: SAM Segmentation ===")
        get_masks_for_bicycle(args.image_dir, args.mask_dir)

        # 2. COLMAP filtering
        print("\n=== Step 2: COLMAP Filtering ===")
        filter_bicycle_from_sfm(args.sparse_in, args.mask_dir, args.sparse_out)

    # 3. Multi-View Inpainting
    print("\n=== Step 3: Multi-View Inpainting ===")
    inpaint_multiview(
        args.image_dir,
        args.mask_dir,
        args.inpaint_out,
        batch_size=int(args.batch_size),
        max_images=int(args.max_images),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        prompt=str(args.prompt),
        negative_prompt=str(args.negative_prompt),
    )
    
    print("\nAll done! Results saved to output directory.")
