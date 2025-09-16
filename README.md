# Digital Image Processing: Traditional Methods for Image Restoration and Segmentation

This repository contains implementations and research on traditional digital image processing methods for image restoration and image segmentation, demonstrating the reliability and effectiveness of classical approaches in computer vision tasks.

## Overview

This project explores two fundamental areas of digital image processing:
1. **Image Enhancement and Restoration** - Focusing on motion blur removal using frequency domain analysis
2. **Image Segmentation and Morphology** - Detecting and separating circular objects using traditional morphological operations

## Project Structure

### Part 1: Image Enhancement and Restoration

The first part addresses the challenge of restoring blurred images, particularly those affected by motion blur. The methodology combines frequency domain analysis with spatial domain enhancement techniques.

#### Key Features:
- **Motion Blur Detection**: Automated identification and classification of blur types
- **Frequency Domain Analysis**: Using Fast Fourier Transform (FFT) for blur characterization
- **Point Spread Function (PSF) Modeling**: Mathematical modeling of blur effects
- **Multiple Restoration Filters**: Implementation of Inverse, Constrained Least Squares, and Wiener filtering
- **Post-processing Enhancement**: Histogram equalization for improved visual quality

#### Technical Approach:
1. Blur type identification and classification
2. Degradation function modeling using FFT analysis
3. PSF creation based on frequency domain characteristics
4. Comparative evaluation of restoration filters
5. Enhancement through histogram equalization

### Part 2: Image Segmentation and Morphology

The second part focuses on detecting and segmenting circular objects of varying sizes, including those touching image boundaries, using traditional morphological operations.

#### Key Features:
- **Boundary Circle Detection**: Identification of circles merged with image edges
- **Multi-scale Circle Detection**: Separate detection of large and small circles
- **Hough Transform Implementation**: Robust circular object detection
- **Morphological Operations**: Advanced boundary delineation techniques
- **Hierarchical Processing**: Systematic approach to complex segmentation tasks

#### Technical Approach:
1. Edge detection using Canny algorithm
2. Contour analysis and minimum enclosing circle calculation
3. Hough Circle Transform for small circle detection
4. Large circle extraction through exclusion methods
5. Boundary delineation using morphological operations (closing, erosion, opening)


## Key Algorithms

### Restoration Methods
- **Fast Fourier Transform (FFT)**: For frequency domain analysis
- **Wiener Filter**: Optimal statistical restoration approach
- **Inverse Filtering**: Direct blur reversal method
- **Constrained Least Squares**: Regularized restoration approach
- **Histogram Equalization**: Contrast enhancement technique

### Segmentation Methods
- **Canny Edge Detection**: Robust edge extraction
- **Hough Circle Transform**: Parameter space circle detection
- **Morphological Operations**: Shape-based image processing
- **Contour Analysis**: Geometric shape approximation
- **Laplacian Operator**: Boundary extraction

## Experimental Results

### Image Restoration Performance
The Wiener filter combined with histogram equalization achieved the highest performance:
- **Mean Gradient Magnitude**: 102.2973 (compared to 12.2187 for original blurred image)
- **Superior noise suppression** compared to inverse and constrained least squares filtering
- **Effective detail preservation** in restored images

### Image Segmentation Accuracy
The morphological approach successfully:
- Detected boundary-touching circles with high precision
- Accurately separated large and small circles
- Effectively delineated boundaries between different circle groups
- Maintained robustness in complex overlapping scenarios

## Applications

This research demonstrates the reliability of traditional methods in:
- **Medical Image Analysis**: Restoration of motion-blurred medical scans
- **Object Detection**: Circular object identification in industrial inspection
- **Computer Vision**: Preprocessing for advanced vision systems
- **Photography**: Enhancement of motion-blurred photographs
- **Autonomous Systems**: Real-time image processing with computational efficiency

## Advantages of Traditional Methods

1. **Computational Efficiency**: Lower resource requirements compared to deep learning approaches
2. **Interpretability**: Clear mathematical foundation and predictable behavior
3. **Reliability**: Consistent performance across different scenarios
4. **Real-time Capability**: Suitable for time-critical applications
5. **No Training Required**: Direct implementation without dataset dependencies

## Dependencies

- OpenCV for image processing operations
- NumPy for numerical computations
- Matplotlib for visualization
- SciPy for advanced mathematical functions

## Future Work

- Extension to non-circular shape detection
- Integration with machine learning for hybrid approaches
- Real-time implementation optimization
- Application to video processing scenarios
- Handling of more complex degradation models

## References

1. Jia, J. Single image motion deblurring using transparency. CVPR 2007.
2. Cooley, J.W.; Tukey, J.W. An algorithm for the machine calculation of complex Fourier series. Mathematics of Computation, 1965.
3. Duda, R.O.; Hart, P.E. Use of the Hough Transformation to Detect Lines and Curves in Pictures. Communications of the ACM, 1972.
4. Canny, J. A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1986.

## License

This project is developed for academic research purposes, demonstrating the effectiveness and reliability of traditional digital image processing methods.

## Author

Name: Pan Yuxuan  
Email: 978299659@qq.com
