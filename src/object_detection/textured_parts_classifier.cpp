
#include "textured_parts_classifier.h"

#include <highgui.h>


using object_detection::TexturedPartsClassifier;

TexturedPartsClassifier::TexturedPartsClassifier() :
    num_bins_(DEFAULT_NUM_BINS)
{
}

cv::Mat TexturedPartsClassifier::preprocessImage(const cv::Mat& image) const
{
    cv::Mat grayscale_image;
    cv::cvtColor(image, grayscale_image, CV_BGR2GRAY);

    // sliding window
    int window_size = 3;
    int border_size = window_size / 2;
    cv::Mat bordered_image;
    cv::copyMakeBorder(grayscale_image, bordered_image, border_size, border_size,
            border_size, border_size, cv::BORDER_REFLECT_101);

    cv::Mat texture_image(image.rows, image.cols, CV_32FC1);

    for(int r = 0; r < texture_image.rows; ++r)
    {
        for(int c = 0; c < texture_image.cols; ++c)
        {
            cv::Rect roi(c, r, window_size, window_size);
            cv::Mat window(bordered_image, roi);
            //double min, max;
            //cv::minMaxLoc(window, &min, &max);
            cv::Scalar mean, stddev;
            cv::meanStdDev(window, mean, stddev);
            texture_image.at<float>(r, c) = stddev[0];
            //texture_image.at<unsigned char>(r, c) = 
            //    bordered_image.at<unsigned char>(r + border_size, c + border_size);
        }
    }
    double min, max;
    cv::minMaxLoc(texture_image, &min, &max);
    texture_image /= max;

    cv::imshow("texture image", texture_image);
    return image;
}

cv::MatND TexturedPartsClassifier::computeHistogram(const cv::Mat& image, 
            const cv::Mat& mask) const
{
    // we assume that the image is a regular
    // three channel image
    CV_Assert(image.type() == CV_8UC3);

    cv::Mat preprocessed_image = preprocessImage(image);

    // dimensions of the histogram
    int histogram_size[] = {num_bins_};

    // ranges for the histogram
    float channel1_ranges[] = {0, 255};
    const float* ranges[] = {channel1_ranges};

    int channels[] = {0};

    cv::MatND histogram;

    // calculation
    int num_arrays = 1;
    int dimensions = 1;
    cv::calcHist(&preprocessed_image, num_arrays, channels, mask, histogram, 
            dimensions, histogram_size, ranges);

    return histogram;

}

cv::Mat TexturedPartsClassifier::backprojectHistogram(const cv::MatND& histogram,
            const cv::Mat& image) const
{
    // we assume that the image is a regular
    // three channel image
    CV_Assert(image.type() == CV_8UC3);

    cv::Mat preprocessed_image = preprocessImage(image);

    int channels[] = {0};

    // ranges for the histogram
    float channel1_ranges[] = {0, 255};
    const float* ranges[] = {channel1_ranges};

    cv::Mat back_projection;
    int num_arrays = 1;
    cv::calcBackProject(&preprocessed_image, num_arrays, channels, histogram,
           back_projection, ranges);

    return back_projection;
}


