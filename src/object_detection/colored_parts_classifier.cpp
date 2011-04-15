
#include "histogram_utilities.h"
#include "colored_parts_classifier.h"


using object_detection::ColoredPartsClassifier;
using object_detection::histogram_utilities::calculateHistogram;
using object_detection::histogram_utilities::calculateBackprojection;

static const int NUM_HUE_BINS = 32;
static const int NUM_SATURATION_BINS = 32;


cv::MatND ColoredPartsClassifier::computeHistogram(const cv::Mat& image, 
            const cv::Mat& mask) const
{
    // convert image to hsv
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, CV_BGR2HSV);

    return calculateHistogram(hsv_image, NUM_HUE_BINS,
            NUM_SATURATION_BINS, mask);
}

cv::Mat ColoredPartsClassifier::backprojectHistogram(const cv::MatND& histogram,
            const cv::Mat& image) const
{
    // convert input image to hsv
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, CV_BGR2HSV);

    // perform back projection
    cv::Mat back_projection = 
        calculateBackprojection(histogram, hsv_image);

    return back_projection;
}


