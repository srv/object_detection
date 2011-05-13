
#include <cassert>
#include <iostream>

#include "glcm.h"


cv::Mat object_detection::computeGLCM(const cv::Mat& image, int dx, int dy, int size)
{
    assert(dx >= 0 && dy >= 0);
    assert(dx != 0 || dy != 0);
    // size must be positive power of two
    assert(size > 1);
    assert((size & (size - 1)) == 0);
    assert(image.channels() == 1);
    assert(image.depth() == CV_8U);

    assert(dx < image.cols);
    assert(dy < image.rows);

    int scale = 256 / size;
    cv::Mat glcm(size, size, CV_32FC(image.channels()), cv::Scalar(0));
    for (int r = 0; r < image.rows - dy; ++r)
    {
        for (int c = 0; c < image.cols - dx; ++c)
        {
            int reference_value = image.at<unsigned char>(r, c) / scale;
            int neighbor_value = image.at<unsigned char>(r + dy, c + dx) / scale;
            glcm.at<float>(reference_value, neighbor_value)+= 1;
        }
    }
    glcm += glcm.t();
    glcm /= 2 * (image.rows - dy) * (image.cols - dx);

    return glcm;
}

cv::Scalar object_detection::computeGLCMFeatures(const cv::Mat& glcm)
{
    assert(glcm.depth() == CV_32F);
    assert(glcm.rows == glcm.cols);
    assert(glcm.channels() == 1);

    double dissimilarity = 0.0;
    double uniformity = 0.0;
    double entropy = 0.0;
    double contrast = 0.0;
    for (int r = 0; r < glcm.rows; ++r)
    {
        for (int c = 0; c < glcm.cols; ++c)
        {
            double value = glcm.at<float>(r, c);
            dissimilarity += value * fabs(c - r);
            uniformity += value * value;
            entropy += value * log(value);
            contrast += value * (r - c) * (r - c);
        }
    }
    return cv::Scalar(dissimilarity, uniformity, entropy, contrast);
}

