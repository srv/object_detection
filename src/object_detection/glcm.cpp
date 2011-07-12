
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

cv::Mat object_detection::computeSlidingWindowUniformGLCM(const cv::Mat& image, int glcm_size, int window_size)
{
    assert(image.channels() == 3);
    assert(image.depth() == CV_8U);

    cv::Mat grayscale_image;
    cv::cvtColor(image, grayscale_image, CV_BGR2GRAY);
 
    // sliding window glcm
    int border_size = window_size / 2;
    cv::Mat bordered_image;
    cv::copyMakeBorder(grayscale_image, bordered_image, border_size, border_size,
            border_size, border_size, cv::BORDER_REFLECT_101);

    cv::Mat texture_image(image.rows, image.cols, CV_32FC4);

    for(int r = 0; r < image.rows; ++r)
    {
        for(int c = 0; c < image.cols; ++c)
        {
            cv::Rect roi(c, r, window_size, window_size);
            cv::Mat window(bordered_image, roi);
            cv::Mat glcm = computeUniformGLCM(window, glcm_size);
            cv::Scalar texture_features = computeGLCMFeatures(glcm);
            texture_image.at<cv::Vec4f>(r, c)[0] = texture_features[0];
            texture_image.at<cv::Vec4f>(r, c)[1] = texture_features[1];
            texture_image.at<cv::Vec4f>(r, c)[2] = texture_features[2];
            texture_image.at<cv::Vec4f>(r, c)[3] = texture_features[3];
        }
    }
    return texture_image;
}

cv::Mat object_detection::computeUniformGLCM(const cv::Mat& image, int size)
{
    // size must be positive power of two
    assert(size > 1);
    assert((size & (size - 1)) == 0);
    assert(image.channels() == 1);
    assert(image.depth() == CV_8U);

    int scale = 256 / size;
    cv::Mat glcm_right(size, size, CV_32FC(image.channels()), cv::Scalar(0));
    cv::Mat glcm_bottom(size, size, CV_32FC(image.channels()), cv::Scalar(0));
    cv::Mat glcm_bottom_right(size, size, CV_32FC(image.channels()), cv::Scalar(0));
    for (int r = 0; r < image.rows - 1; ++r)
    {
        for (int c = 0; c < image.cols - 1; ++c)
        {
            int reference_value = image.at<unsigned char>(r, c) / scale;
            int right_value = image.at<unsigned char>(r, c + 1) / scale;
            int bottom_value = image.at<unsigned char>(r + 1, c) / scale;
            int bottom_right_value = image.at<unsigned char>(r + 1, c + 1) / scale;
            glcm_right.at<float>(reference_value, right_value)+= 1;
            glcm_bottom.at<float>(reference_value, bottom_value)+= 1;
            glcm_bottom_right.at<float>(reference_value, bottom_right_value)+= 1;
        }
    }
    cv::Mat glcm(size, size, CV_32FC(image.channels()), cv::Scalar(0));
    glcm += glcm_right;
    glcm += glcm_right.t();
    glcm += glcm_bottom;
    glcm += glcm_bottom.t();
    glcm += glcm_bottom_right;
    glcm += glcm_bottom_right.t();
    glcm /= 6 * (image.rows - 1) * (image.cols - 1);

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
    double glcm_mean = 0.0;
    for (int r = 0; r < glcm.rows; ++r)
    {
        for (int c = 0; c < glcm.cols; ++c)
        {
            double value = glcm.at<float>(r, c);
            dissimilarity += value * fabs(c - r);
            uniformity += value * value;
            if (value != 0) entropy += value * log(value);
            glcm_mean += r * value;
        }
    }
    entropy = -entropy; 
    return cv::Scalar(dissimilarity, uniformity, entropy, glcm_mean);
}
