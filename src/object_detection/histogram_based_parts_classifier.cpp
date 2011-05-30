#include <string>
#include <vector>
#include <stdexcept>

#include <cv.h>

#include "histogram_based_parts_classifier.h"
#include "histogram_utilities.h"

using object_detection::HistogramBasedPartsClassifier; 
using object_detection::histogram_utilities::showHistogram;
using object_detection::histogram_utilities::showHSHistogram;

HistogramBasedPartsClassifier::HistogramBasedPartsClassifier()
    : min_occurences_(0)
{
}

void HistogramBasedPartsClassifier::setMinOccurences(int min_occurences)
{
    assert(min_occurences >= 0);
    min_occurences_ = min_occurences;
}

int HistogramBasedPartsClassifier::minOccurences() const
{
    return min_occurences_;
}

/*
void HistogramBasedPartsClassifier::train(const cv::Mat& image, const cv::Mat& mask)
{
    // compute histograms of foreground and background and divide them
    // (virtual function calls)
    cv::MatND object_histogram = computeHistogram(image, mask);
    cv::MatND background_histogram = computeHistogram(image, 255 - mask);
    significant_elements_histogram_ = object_histogram / (background_histogram + 1);

    showHistogram(object_histogram, getName() + "-object histogram");
    showHistogram(background_histogram, getName() + "-background histogram");
    showHistogram(significant_elements_histogram_, getName() + "-significant elements histogram");
}
*/
 
void HistogramBasedPartsClassifier::train(const cv::Mat& image, const cv::Mat& mask)
{
    // compute histograms of object and whole image and divide them
    cv::MatND object_histogram = computeHistogram(image, mask);
    cv::MatND image_histogram = computeHistogram(image, cv::Mat());

    // filter out colors with few occurences
    cv::threshold(object_histogram, object_histogram, min_occurences_, 255, CV_THRESH_TOZERO);

    significant_elements_histogram_ = cv::MatND(object_histogram / image_histogram);

    showHSHistogram(object_histogram, getName() + "-object histogram");
    showHSHistogram(image_histogram, getName() + "-image histogram");
    showHSHistogram(significant_elements_histogram_, getName() + "-significant elements histogram");
}
   
cv::Mat HistogramBasedPartsClassifier::classify(const cv::Mat& image, 
            const std::vector<cv::Rect>& rois) const
{
    assert(significant_elements_histogram_.size[0] != 0);
    if (significant_elements_histogram_.size[0] == 0)
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

    // some opening
    int element_size = 7;
    cv::Mat element = cv::Mat::zeros(element_size, element_size, CV_8UC1);
    cv::circle(element, cv::Point(element_size / 2, element_size / 2), element_size / 2, cv::Scalar(255), -1);
    //cv::morphologyEx(probability_image, probability_image, cv::MORPH_CLOSE, element);
    cv::morphologyEx(probability_image, probability_image, cv::MORPH_OPEN, element);

    
    return probability_image;
}


