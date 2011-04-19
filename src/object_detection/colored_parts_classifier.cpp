
#include "histogram_utilities.h"
#include "colored_parts_classifier.h"

#include <highgui.h>


using object_detection::ColoredPartsClassifier;
using object_detection::histogram_utilities::calculateHistogram;
using object_detection::histogram_utilities::calculateBackprojection;

static const int NUM_HUE_BINS = 32;
static const int NUM_SATURATION_BINS = 16;

cv::Mat ColoredPartsClassifier::preprocessImage(const cv::Mat& image)
{
    // convert image to hsv
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, CV_BGR2HSV);
    return hsv_image;
}

cv::MatND ColoredPartsClassifier::computeHistogram(const cv::Mat& image, 
            const cv::Mat& mask) const
{
    cv::Mat preprocessed_image = preprocessImage(image);
    return calculateHistogram(preprocessed_image, NUM_HUE_BINS,
            NUM_SATURATION_BINS, mask);
}

cv::Mat ColoredPartsClassifier::backprojectHistogram(const cv::MatND& histogram,
            const cv::Mat& image) const
{
    cv::Mat preprocessed_image = preprocessImage(image);

    // perform back projection
    cv::Mat back_projection = 
        calculateBackprojection(histogram, preprocessed_image);

    return back_projection;
}


