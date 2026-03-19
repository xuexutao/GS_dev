from plyfile import PlyData


def main() -> None:
    path = "output/bicycle_obj0002/point_cloud/iteration_7000/point_cloud.ply"
    ply = PlyData.read(path)
    print("ply:", path)
    print("vertices:", ply["vertex"].count)


if __name__ == "__main__":
    main()

