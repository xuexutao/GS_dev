#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
import glob
from PIL import Image
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, mask_opt=opt)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    def _filter_views_with_masks(views):
        if not getattr(opt, "mask_only", False):
            return views
        # Keep only views that have required masks on disk.
        kept = []
        for v in views:
            stem = os.path.splitext(os.path.basename(v.image_name))[0]
            mask_dir = os.path.join(dataset.source_path, dataset.images, opt.mask_dirname, stem)
            if int(getattr(opt, "mask_object_id", -1)) >= 0:
                mid = int(opt.mask_object_id)
                if os.path.exists(os.path.join(mask_dir, f"obj_{mid:04d}.png")):
                    kept.append(v)
            else:
                if os.path.isdir(mask_dir) and len(glob.glob(os.path.join(mask_dir, "*.png"))) > 0:
                    kept.append(v)
        if not kept:
            raise RuntimeError(
                "mask_only is enabled but no views have masks on disk. "
                "Run generate_multiview_sam_masks.py first, and ensure --images matches where masks are written."
            )
        return kept

    viewpoint_stack = _filter_views_with_masks(viewpoint_stack)
    viewpoint_indices = list(range(len(viewpoint_stack)))

    mask_cache = {}

    def _load_object_masks_for_view(viewpoint_cam):
        """Load list of object masks (1,H,W) in [0,1], resized to current training resolution."""
        key = (viewpoint_cam.image_name, int(viewpoint_cam.image_height), int(viewpoint_cam.image_width))
        if key in mask_cache:
            return mask_cache[key]

        stem = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
        mask_dir = os.path.join(dataset.source_path, dataset.images, opt.mask_dirname, stem)
        if not os.path.isdir(mask_dir):
            mask_cache[key] = []
            return []

        if int(getattr(opt, "mask_object_id", -1)) >= 0:
            mid = int(opt.mask_object_id)
            paths = [os.path.join(mask_dir, f"obj_{mid:04d}.png")]
        else:
            paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        if not paths:
            mask_cache[key] = []
            return []

        # Load masks as float tensor on CPU first
        masks = []
        H, W = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)
        for p in paths[: int(opt.mask_max_objects)]:
            if not os.path.exists(p):
                continue
            m = Image.open(p).convert("L")
            m = torch.from_numpy(np.array(m, dtype=np.uint8)).float() / 255.0
            if m.shape[0] != H or m.shape[1] != W:
                m = F.interpolate(m[None, None, ...], size=(H, W), mode="nearest")[0, 0]
            masks.append(m[None, ...])

        if not masks:
            mask_cache[key] = []
            return []

        # Filter tiny masks
        out = []
        total = float(H * W)
        for m in masks:
            cov = float((m > 0.5).sum().item()) / max(total, 1.0)
            if cov >= float(opt.mask_min_coverage):
                out.append(m)
        mask_cache[key] = out
        return out

    def _masked_l1(img, gt, mask_1hw):
        # img/gt: (3,H,W), mask: (1,H,W)
        mask3 = mask_1hw.expand_as(img)
        diff = torch.abs(img - gt) * mask3
        denom = mask3.sum().clamp_min(1.0)
        return diff.sum() / denom

    def _packed_mask_loss(img, gt, masks_1hw, pack_size: int):
        """Pack multiple masked crops into a pack_size x pack_size atlas and compute L1."""
        n = len(masks_1hw)
        if n == 0:
            return None
        grid = int(np.ceil(np.sqrt(n)))
        tile = max(int(pack_size // grid), 1)
        atlas_img = torch.zeros((3, pack_size, pack_size), device=img.device, dtype=img.dtype)
        atlas_gt = torch.zeros((3, pack_size, pack_size), device=gt.device, dtype=gt.dtype)

        placed = 0
        for i, m in enumerate(masks_1hw):
            if placed >= grid * grid:
                break
            ys, xs = torch.where(m[0] > 0.5)
            if ys.numel() == 0:
                continue
            y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
            x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
            # crop and apply mask
            crop_m = m[:, y0:y1, x0:x1]
            crop_img = img[:, y0:y1, x0:x1] * crop_m
            crop_gt = gt[:, y0:y1, x0:x1] * crop_m

            crop_img = F.interpolate(crop_img[None], size=(tile, tile), mode="bilinear", align_corners=False)[0]
            crop_gt = F.interpolate(crop_gt[None], size=(tile, tile), mode="bilinear", align_corners=False)[0]

            r = placed // grid
            c = placed % grid
            oy, ox = r * tile, c * tile
            atlas_img[:, oy : oy + tile, ox : ox + tile] = crop_img
            atlas_gt[:, oy : oy + tile, ox : ox + tile] = crop_gt
            placed += 1

        if placed == 0:
            return None
        return torch.abs(atlas_img - atlas_gt).mean()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # ======================================================
    # 迭代训练
    # ======================================================
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record() # 记录当前迭代开始时间

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = _filter_views_with_masks(scene.getTrainCameras().copy())
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # If mask_only is enabled and we want to use the masked rasterizer, avoid the unmasked render
        # (it may include verbose CUDA debug prints in some builds, and is unnecessary for gradients).
        if getattr(opt, "mask_only", False) and opt.mask_use_masked_rasterizer:
            obj_masks = _load_object_masks_for_view(viewpoint_cam)
            if not obj_masks:
                raise RuntimeError(
                    f"mask_only is enabled but no masks found for view {viewpoint_cam.image_name}. "
                    f"Expected: <source>/<images>/{opt.mask_dirname}/{os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]}/obj_*.png"
                )
            obj_masks = [m.to(device=bg.device) for m in obj_masks]
            union_mask = torch.clamp(torch.stack(obj_masks, dim=0).sum(dim=0), 0.0, 1.0)
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                mask=union_mask,
                use_masked_rasterizer=True,
                apply_mask_in_forward=False,
            )
        else:
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # Optional: object-only training (only optimize pixels inside selected mask)
        if getattr(opt, "mask_only", False):
            # When opt.mask_use_masked_rasterizer is enabled, `render_pkg` above already used it.
            obj_masks = _load_object_masks_for_view(viewpoint_cam)
            if not obj_masks:
                raise RuntimeError(
                    f"mask_only is enabled but no masks found for view {viewpoint_cam.image_name}. "
                    f"Expected: <source>/<images>/{opt.mask_dirname}/{os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]}/obj_*.png"
                )
            obj_masks = [m.to(device=image.device) for m in obj_masks]
            union_mask = torch.clamp(torch.stack(obj_masks, dim=0).sum(dim=0), 0.0, 1.0)
            mask3 = union_mask.expand_as(image)

            Ll1 = _masked_l1(image, gt_image, union_mask)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(
                    (image * mask3).unsqueeze(0),
                    (gt_image * mask3).unsqueeze(0),
                )
            else:
                ssim_value = ssim(image * mask3, gt_image * mask3)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        else:
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Mask loss (optional): full-image loss + object-mask losses
            if opt.mask_loss_weight > 0.0:
                obj_masks = _load_object_masks_for_view(viewpoint_cam)
                if obj_masks:
                    obj_masks = [m.to(device=image.device) for m in obj_masks]
                    union_mask = torch.clamp(torch.stack(obj_masks, dim=0).sum(dim=0), 0.0, 1.0)

                    # Per-object masked loss on current rendered image
                    per_obj = 0.0
                    for m in obj_masks:
                        per_obj = per_obj + _masked_l1(image, gt_image, m)
                    per_obj = per_obj / float(len(obj_masks))

                    # Optional: run a second render path that uses masked rasterizer (mask applied in forward/backward)
                    if opt.mask_use_masked_rasterizer:
                        render_pkg_masked = render(
                            viewpoint_cam,
                            gaussians,
                            pipe,
                            bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            mask=union_mask,
                            use_masked_rasterizer=True,
                            apply_mask_in_forward=True,
                        )
                        image_masked = render_pkg_masked["render"]
                        union_obj = _masked_l1(image_masked, gt_image, union_mask)
                    else:
                        union_obj = _masked_l1(image, gt_image, union_mask)

                    if opt.mask_pack_loss:
                        packed = _packed_mask_loss(image, gt_image, obj_masks, int(opt.mask_pack_size))
                    else:
                        packed = None

                    if packed is None:
                        mask_loss = 0.5 * union_obj + 0.5 * per_obj
                    else:
                        mask_loss = (union_obj + per_obj + packed) / 3.0
                    loss = loss + float(opt.mask_loss_weight) * mask_loss

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
