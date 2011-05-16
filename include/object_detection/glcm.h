#ifndef GLCM_H
#define GLCM_H

#include <cv.h>

namespace object_detection {

    /**
    * Computes the glcm for the given image.
    * The matrix will be symmetric and normalized.
    */
    cv::Mat computeGLCM(const cv::Mat& image, int dx, int dy, int size = 256);

    /**
    * Computes the uniform glcm for the given image.
    * Tree glcm will be computed for the directions right, bottom and bottom 
    * right. The final glcm will be computed as the mean of these.
    * The matrix will be symmetric and normalized.
    */
    cv::Mat computeUniformGLCM(const cv::Mat& image, int size = 256);

    /**
    * computes the four glcm features
    * -# dissimilarity
    * -# uniformity
    * -# entropy
    * -# glcm_mean
    * The values are stored in that order in a cv::Scalar and returned.
    */
    cv::Scalar computeGLCMFeatures(const cv::Mat& glcm);

} // namespace object_detection

#endif

