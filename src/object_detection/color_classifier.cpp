
#include <iostream>
#include <stdexcept>
#include <highgui.h>

#include "color_classifier.h" 
#include "histogram_utilities.h"
#include "detection.h"
#include "training_data.h"
#include "utilities.h"


using object_detection::ColorClassifier;
using object_detection::histogram_utilities::calculateHistogram;
using object_detection::histogram_utilities::calculateBackprojection;
using object_detection::histogram_utilities::accumulateHistogram;

static const int NUM_HUE_BINS = 32;
static const int NUM_SATURATION_BINS = 32;

ColorClassifier::ColorClassifier() : is_trained_(false)
{
}

void ColorClassifier::train(const cv::Mat& image, const cv::Mat& mask)
{
    // convert image to hsv
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, CV_BGR2HSV);

    accumulateHistogram(hsv_image, NUM_HUE_BINS,
            NUM_SATURATION_BINS, mask, object_histogram_);

    cv::Mat background_mask = 255 - mask;
    accumulateHistogram(hsv_image, NUM_HUE_BINS,
            NUM_SATURATION_BINS, background_mask, background_histogram_);

    // divide the histograms so that only those colors remain
    // that are representative for the object
    significant_colors_histogram_ = object_histogram_ / (background_histogram_ + 1);
    is_trained_ = true;
 }

cv::Mat ColorClassifier::classify(const cv::Mat& image,
        const std::vector<cv::Rect>& rois) const
{
    if (!is_trained_)
    {
        throw std::runtime_error("ColorClassifier::classify() called without having trained before");
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

    // convert input image to hsv
    cv::Mat hsv_image;
    cv::cvtColor(image_masked, hsv_image, CV_BGR2HSV);

    // perform back projection
    cv::Mat back_projection = 
        calculateBackprojection(significant_colors_histogram_, hsv_image);

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

