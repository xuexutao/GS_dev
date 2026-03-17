from typing import NamedTuple, Optional

import torch
import torch.nn as nn


try:
    # Reuse the compiled C++/CUDA extension from the original rasterizer.
    from diff_gaussian_rasterization import _C  # type: ignore
except Exception as e:  # pragma: no cover
    _C = None
    _IMPORT_ERROR = e


def _expand_mask(mask: torch.Tensor, color_like: torch.Tensor) -> torch.Tensor:
    """Expand mask to match CHW tensor (float32/float16), values assumed in [0,1]."""
    if mask.dim() == 2:
        mask = mask[None, ...]
    if mask.dim() == 3 and mask.shape[0] == 1 and color_like.dim() == 3 and color_like.shape[0] == 3:
        mask = mask.expand(3, -1, -1)
    return mask


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    antialiasing: bool


class _RasterizeGaussiansMasked(torch.autograd.Function):
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
        if _C is None:  # pragma: no cover
            raise ImportError(
                "diff_gaussian_rasterization is not available; cannot use masked rasterizer"
            ) from _IMPORT_ERROR

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
        )

        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(
            *args
        )

        if mask is not None and apply_mask_in_forward:
            m = _expand_mask(mask, color)
            color = color * m
            invdepths = invdepths * (m[0:1, ...] if invdepths.dim() == 3 else 1.0)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.apply_mask_in_backward = mask is not None
        ctx.mask = mask
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):
        if _C is None:  # pragma: no cover
            raise ImportError(
                "diff_gaussian_rasterization is not available; cannot use masked rasterizer"
            ) from _IMPORT_ERROR

        raster_settings = ctx.raster_settings
        num_rendered = ctx.num_rendered
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = (
            ctx.saved_tensors
        )

        if ctx.apply_mask_in_backward and ctx.mask is not None:
            m = _expand_mask(ctx.mask, grad_out_color)
            grad_out_color = grad_out_color * m
            if isinstance(grad_out_depth, torch.Tensor):
                if grad_out_depth.dim() == 3:
                    grad_out_depth = grad_out_depth * m[0:1, ...]
                else:
                    grad_out_depth = grad_out_depth

        args = (
            raster_settings.bg,
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
        )

        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = (
            _C.rasterize_gaussians_backward(*args)
        )

        return (
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
    return _RasterizeGaussiansMasked.apply(
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


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            return _C.mark_visible(positions, raster_settings.viewmatrix, raster_settings.projmatrix)

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
            raise Exception("Please provide excatly one of either SHs or precomputed colors!")

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

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

