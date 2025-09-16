import argparse
from image_restoration import ImageRestoration
from image_segmentation import ImageSegmentation


def main():
    parser = argparse.ArgumentParser(description='图像处理项目：复原与分割')
    parser.add_argument('--restoration', action='store_true', help='运行图像复原')
    parser.add_argument('--segmentation', action='store_true', help='运行图像分割')
    parser.add_argument('--all', action='store_true', help='运行所有处理')

    args = parser.parse_args()

    if args.all or (not args.restoration and not args.segmentation):
        # 默认运行所有处理
        args.restoration = True
        args.segmentation = True

    if args.restoration:
        print("=" * 50)
        print("开始图像复原处理")
        print("=" * 50)
        ImageRestoration.run_restoration()

    if args.segmentation:
        print("=" * 50)
        print("开始图像分割处理")
        print("=" * 50)
        ImageSegmentation.run_segmentation()


if __name__ == "__main__":
    main()