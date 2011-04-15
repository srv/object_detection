#include <string>
#include <vector>

#include <cv.h>

#include "histogram_based_parts_classifier.h"

using object_detection::HistogramBasedPartsClassifier; 


void HistogramBasedPartsClassifier::train(const cv::Mat& image, const cv::Mat& mask)
{
    // compute histograms of foreground and background and divide them
    // (virtual function calls)
    cv::MatND object_histogram = computeHistogram(image, mask);
    cv::MatND background_histogram = computeHistogram(image, 255 - mask);
    significant_elements_histogram_ = object_histogram / (background_histogram + 1);
}
    
cv::Mat HistogramBasedPartsClassifier::classify(const cv::Mat& image, 
            const std::vector<cv::Rect>& rois) const
{
    assert(!significant_elements_histogram_.empty());
    if (significant_elements_histogram_.empty())
    {
        throw std::runtime_error("HistogramBasedPartsClassifier::classify() called without having trained before");
    }

    // mask out anything outside input ROIs
    cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    if (rois.size() > 0)
    {
        for (size_t i = 0; i < rois.size(); ++i)
        {
            cv::rectangle(mask, rois[i], cv::Scalar(255), CV_FILLED);
        }
    }
    else
    {
        // no roi given, set all to 255
        mask = cv::Scalar(255);
    }

    cv::Mat image_masked;
    image.copyTo(image_masked, mask);

    // perform back projection (virtual function call)
    cv::Mat back_projection = 
        backprojectHistogram(significant_elements_histogram_, image_masked);

    cv::Mat probability_image;
    back_projection.convertTo(probability_image, CV_32F, 1.0/255.0);

    // some closing
    int element_size = 3;
    cv::Mat element = cv::Mat::zeros(element_size, element_size, CV_8UC1);
    cv::circle(element, cv::Point(element_size / 2, element_size / 2), element_size / 2, cv::Scalar(255), -1);
    cv::morphologyEx(probability_image, probability_image, cv::MORPH_CLOSE, element);
    cv::morphologyEx(probability_image, probability_image, cv::MORPH_OPEN, element);

    return probability_image;
}


