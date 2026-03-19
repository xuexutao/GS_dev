import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _src(p: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), p)


def _diff_src(p: str) -> str:
    # Reuse the upstream sources to avoid duplicating files.
    return _src(os.path.join("..", "diff-gaussian-rasterization", p))


glm_include = _src(os.path.join("..", "diff-gaussian-rasterization", "third_party", "glm"))

setup(
    name="masked-diff-gaussian-rasterization",
    version="0.0.3",
    description=(
        "Mask-aware rasterizer built from diff-gaussian-rasterization sources (mask applied in forward/backward)"
    ),
    packages=["masked_diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="masked_diff_gaussian_rasterization._C",
            sources=[
                _diff_src("cuda_rasterizer/rasterizer_impl.cu"),
                _diff_src("cuda_rasterizer/forward.cu"),
                _diff_src("cuda_rasterizer/backward.cu"),
                _diff_src("rasterize_points.cu"),
                _diff_src("ext.cpp"),
            ],
            extra_compile_args={"nvcc": ["-I" + glm_include]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
