import torch


def main() -> None:
    import masked_diff_gaussian_rasterization as m

    print("masked_diff_gaussian_rasterization imported")
    print("has _C:", hasattr(m, "_C"))
    print("cuda available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()

