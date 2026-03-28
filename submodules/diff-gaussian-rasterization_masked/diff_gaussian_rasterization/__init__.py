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

from typing import NamedTuple, Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import _C


def _expand_mask(mask: torch.Tensor, color_like: torch.Tensor) -> torch.Tensor:
    """Expand mask to match CHW tensor.

    - mask can be (H,W) or (1,H,W)
    - if color_like is (3,H,W), expand to (3,H,W)
    """
    if mask.dim() == 2:
        mask = mask[None, ...]
    if (
        mask.dim() == 3
        and mask.shape[0] == 1
        and color_like.dim() == 3
        and color_like.shape[0] == 3
    ):
        mask = mask.expand(3, -1, -1)
    return mask

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    mask: Optional[torch.Tensor] = None,
    apply_mask_in_forward: bool = True,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        mask,
        apply_mask_in_forward,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        mask: Optional[torch.Tensor],
        apply_mask_in_forward: bool,
    ):

        if not means3D.is_cuda:
            raise ValueError("diff_gaussian_rasterization_masked 需要 CUDA tensor（means3D 必须在 CUDA 上）")

        # Canonicalize mask for CUDA/C++: float32, CUDA, contiguous, and 2D (H, W).
        # NOTE: CUDA kernels expect a single plane mask indexed by pix_id = W*y + x.
        tile_mask = None
        mask_cuda = None
        if mask is not None:
            m = mask
            if m.dim() == 3:
                # Accept (1,H,W) or (C,H,W); CUDA side consumes a single plane.
                m = m[0]
            if m.dim() != 2:
                raise ValueError(
                    f"mask 期望 shape 为 (H,W) 或 (1,H,W)/(C,H,W)，但得到 {tuple(mask.shape)}"
                )
            H, W = int(raster_settings.image_height), int(raster_settings.image_width)
            if int(m.shape[0]) != H or int(m.shape[1]) != W:
                raise ValueError(
                    f"mask 的空间尺寸必须与渲染分辨率一致：mask={tuple(m.shape)} vs (H,W)=({H},{W})"
                )
            mask_cuda = m.to(dtype=torch.float32, device=means3D.device).contiguous()

            # Tile-level occupancy mask (for accelerating binning/sort). Must match CUDA BLOCK_X/BLOCK_Y.
            # config.h: BLOCK_X=16, BLOCK_Y=16
            tile_mask = F.max_pool2d(
                mask_cuda[None, None, ...],
                kernel_size=(16, 16),
                stride=(16, 16),
                ceil_mode=True,
            )[0, 0].contiguous()

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
            mask_cuda if mask_cuda is not None else torch.empty(0, dtype=torch.float32, device=means3D.device),
            tile_mask if tile_mask is not None else torch.empty(0, dtype=torch.float32, device=means3D.device),
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # Optional: apply mask in forward (output is masked for downstream losses/visualization)
        if mask_cuda is not None and apply_mask_in_forward:
            m = _expand_mask(mask_cuda, color)
            color = color * m
            if isinstance(invdepths, torch.Tensor) and invdepths.dim() == 3:
                invdepths = invdepths * m[0:1, ...]

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.apply_mask_in_backward = mask_cuda is not None
        ctx.mask = mask_cuda
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Optional: apply mask to gradients (masked loss only backprops through masked pixels)
        if getattr(ctx, "apply_mask_in_backward", False) and getattr(ctx, "mask", None) is not None:
            m = _expand_mask(ctx.mask, grad_out_color)
            grad_out_color = grad_out_color * m
            if isinstance(grad_out_depth, torch.Tensor) and grad_out_depth.dim() == 3:
                grad_out_depth = grad_out_depth * m[0:1, ...]

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug,
                ctx.mask.contiguous() if getattr(ctx, "mask", None) is not None else torch.empty(0, dtype=torch.float32, device=means3D.device))

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        mask: Optional[torch.Tensor] = None,
        apply_mask_in_forward: bool = True,
    ):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings,
            mask=mask,
            apply_mask_in_forward=apply_mask_in_forward,
        )
