import numpy as np
import cv2
import matplotlib.pyplot as plt


class ImageRestoration:
    """图像复原处理类"""

    @staticmethod
    def get_motion_psf(img, angle, dist):
        """生成运动模糊的点扩散函数(PSF)"""
        x_center = img.shape[0] // 2
        y_center = img.shape[1] // 2
        sin_val = np.sin(angle * np.pi / 180)
        cos_val = np.cos(angle * np.pi / 180)
        PSF = np.zeros(img.shape[:2])

        for i in range(dist):
            x_offset = round(sin_val * i)
            y_offset = round(cos_val * i)
            if (x_center - x_offset >= 0 and x_center - x_offset < img.shape[0] and
                    y_center + y_offset >= 0 and y_center + y_offset < img.shape[1]):
                PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1
        return PSF / PSF.sum()

    @staticmethod
    def wiener_filter(img, PSF, eps, K):
        """维纳滤波"""
        fft_img = np.fft.fft2(img)
        fft_PSF = np.fft.fft2(PSF) + eps
        fft_wiener = np.conj(fft_PSF) / (np.abs(fft_PSF) ** 2 + K)
        img_wiener_filter = np.fft.ifft2(fft_img * fft_wiener)
        img_wiener_filter = np.abs(np.fft.fftshift(img_wiener_filter))
        return img_wiener_filter

    @staticmethod
    def inverse_filter(img, PSF, eps=1e-1):
        """逆滤波"""
        fft_img = np.fft.fft2(img)
        fft_PSF = np.fft.fft2(PSF) + eps
        img_inverse_filter = np.fft.ifft2(fft_img / fft_PSF)
        img_inverse_filter = np.abs(np.fft.fftshift(img_inverse_filter))
        return img_inverse_filter

    @staticmethod
    def least_squares_filter(img, PSF, eps=1e-3):
        """最小二乘滤波（伪逆滤波）"""
        fft_img = np.fft.fft2(img)
        fft_PSF = np.fft.fft2(PSF) + eps
        fft_ls = np.conj(fft_PSF) / (np.abs(fft_PSF) ** 2 + eps)
        img_least_squares = np.fft.ifft2(fft_img * fft_ls)
        img_least_squares = np.abs(np.fft.fftshift(img_least_squares))
        return img_least_squares

    @staticmethod
    def histogram_equalization(img):
        """直方图均衡化"""
        return cv2.equalizeHist(img)

    @staticmethod
    def calculate_sharpness(image):
        """计算图像清晰度（基于梯度）"""
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        return np.mean(gradient_magnitude)

    @staticmethod
    def evaluate_image_quality(image, title):
        """评估图像质量"""
        sharpness = ImageRestoration.calculate_sharpness(image)
        print(f"{title} - 清晰度: {sharpness:.4f}")
        return sharpness

    @staticmethod
    def run_restoration():
        """运行图像复原处理"""
        # 读取模糊图像
        blurred_image = cv2.imread("data/boy-blurred.tif", 0)

        # 模糊方向和长度
        angle = 49  # 模糊方向（49度）
        length = 91  # 模糊长度（91像素）

        # 生成PSF
        PSF = ImageRestoration.get_motion_psf(blurred_image, angle, length)

        # 应用逆滤波
        img_inverse_filter = ImageRestoration.inverse_filter(blurred_image, PSF)

        # 应用最小二乘滤波
        img_least_squares = ImageRestoration.least_squares_filter(blurred_image, PSF)

        # 应用维纳滤波
        img_wiener_filter = ImageRestoration.wiener_filter(blurred_image, PSF, eps=1e-3, K=0.000005)

        # 归一化维纳滤波图像
        img_wiener_filter = cv2.normalize(img_wiener_filter, None, 0, 255, cv2.NORM_MINMAX)
        img_wiener_filter = img_wiener_filter.astype(np.uint8)

        # 对维纳滤波图像应用直方图均衡化
        img_wiener_equalized = ImageRestoration.histogram_equalization(img_wiener_filter)

        # 确保所有图像都是uint8类型
        img_inverse_filter = cv2.normalize(img_inverse_filter, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_least_squares = cv2.normalize(img_least_squares, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 评估图像质量
        ImageRestoration.evaluate_image_quality(blurred_image, "模糊图像")
        ImageRestoration.evaluate_image_quality(img_inverse_filter, "逆滤波图像")
        ImageRestoration.evaluate_image_quality(img_least_squares, "最小二乘滤波图像")
        ImageRestoration.evaluate_image_quality(img_wiener_filter, "维纳滤波图像")
        ImageRestoration.evaluate_image_quality(img_wiener_equalized, "维纳滤波+直方图均衡化图像")

        # 显示图像
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(blurred_image, cmap='gray')
        plt.title('模糊图像')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(img_inverse_filter, cmap='gray')
        plt.title('逆滤波')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(img_least_squares, cmap='gray')
        plt.title('最小二乘滤波')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(img_wiener_filter, cmap='gray')
        plt.title('维纳滤波')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(img_wiener_equalized, cmap='gray')
        plt.title('维纳滤波+直方图均衡化')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return {
            'blurred': blurred_image,
            'inverse': img_inverse_filter,
            'least_squares': img_least_squares,
            'wiener': img_wiener_filter,
            'wiener_equalized': img_wiener_equalized
        }