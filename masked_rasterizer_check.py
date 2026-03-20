import torch
import inspect


def main() -> None:
    import diff_gaussian_rasterization as m

    print("diff_gaussian_rasterization imported")
    print("has _C:", hasattr(m, "_C"))
    print("cuda available:", torch.cuda.is_available())
    try:
        sig = inspect.signature(m.GaussianRasterizer.forward)
        print("GaussianRasterizer.forward signature:", sig)
    except Exception as e:
        print("failed to inspect signature:", e)


if __name__ == "__main__":
    main()
