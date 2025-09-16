import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageSegmentation:
    """图像分割处理类"""

    @staticmethod
    def detect_edge_touching_circles(img_path):
        """检测接触边界的圆"""
        img_original = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        img_denoised = cv2.medianBlur(img_gray, 5)

        edges = cv2.Canny(img_denoised, threshold1=30, threshold2=100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        edge_touching_circles = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
            if len(approx) > 8:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                if (x - radius < 0 or x + radius >= img_gray.shape[1] or
                        y - radius < 0 or y + radius >= img_gray.shape[0]):
                    edge_touching_circles.append((center, radius))
                    cv2.circle(img_original, center, 2, (0, 255, 0), 3)
                    cv2.circle(img_original, center, radius, (0, 0, 255), 2)

        plt.figure(figsize=(15, 5))
        images = [img_gray, edges, img_original]
        titles = ['原始图像', 'Canny边缘', '接触边界的圆']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) if i != 1 else images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        return edge_touching_circles

    @staticmethod
    def detect_small_circles(img_path):
        """检测小圆"""
        img_original = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        img_denoised = cv2.medianBlur(img_gray, 5)

        circles = cv2.HoughCircles(img_denoised, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=100, param2=50, minRadius=20, maxRadius=30)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x, y, r = i[0], i[1], i[2]
                if (x - r >= 0 and x + r < img_gray.shape[1] and
                        y - r >= 0 and y + r < img_gray.shape[0]):
                    cv2.circle(img_original, (x, y), 2, (0, 255, 0), 3)
                    cv2.circle(img_original, (x, y), r, (0, 0, 255), 2)

        plt.figure(figsize=(15, 5))
        images = [img_gray, img_denoised, img_original]
        titles = ['原始图像', '去噪图像（中值滤波）', '检测到的小圆']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) if i != 1 else images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        return circles

    @staticmethod
    def morphological_operations(img_path):
        """形态学操作"""
        I = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 闭操作去除内部小圆
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 45))
        K = cv2.morphologyEx(I, cv2.MORPH_CLOSE, se)

        # 腐蚀使大圆区域更突出
        se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        L = cv2.erode(K, se3)

        # 开操作去除噪声
        se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
        J = cv2.morphologyEx(L, cv2.MORPH_OPEN, se1)

        # 二值化图像
        _, J = cv2.threshold(J, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 提取边界
        M = cv2.Laplacian(J, cv2.CV_64F)
        M = np.uint8(np.absolute(M))

        # 转换为彩色图像
        I_color = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)

        # 用红色叠加边界
        red_color = (0, 0, 255)
        I_color[M > 0] = red_color

        # 显示结果
        plt.figure(figsize=(12, 8))

        plt.subplot(231), plt.imshow(I, cmap='gray'), plt.title('原始图像')
        plt.axis('off')

        plt.subplot(232), plt.imshow(K, cmap='gray'), plt.title('闭操作')
        plt.axis('off')

        plt.subplot(233), plt.imshow(L, cmap='gray'), plt.title('腐蚀')
        plt.axis('off')

        plt.subplot(234), plt.imshow(J, cmap='gray'), plt.title('开操作和二值化')
        plt.axis('off')

        plt.subplot(235), plt.imshow(cv2.cvtColor(I_color, cv2.COLOR_BGR2RGB)), plt.title('绘制边界')
        plt.axis('off')

        plt.show()

        return I_color

    @staticmethod
    def run_segmentation():
        """运行图像分割处理"""
        print("处理 circles.tif 图像...")
        edge_circles = ImageSegmentation.detect_edge_touching_circles('data/circles.tif')
        print(f"检测到 {len(edge_circles)} 个接触边界的圆")

        print("\n处理 FigP0934.tif 图像...")
        small_circles = ImageSegmentation.detect_small_circles('data/FigP0934.tif')
        print(f"检测到 {len(small_circles[0]) if small_circles is not None else 0} 个小圆")

        print("\n应用形态学操作...")
        result_img = ImageSegmentation.morphological_operations('data/FigP0934.tif')

        return {
            'edge_circles': edge_circles,
            'small_circles': small_circles,
            'morphological_result': result_img
        }