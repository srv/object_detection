
#include <iostream>
#include <stdexcept>
#include <highgui.h>

#include "histogram_backprojection.h" 
#include "histogram_utilities.h"
#include "detection.h"
#include "training_data.h"
#include "utilities.h"


using object_detection::HistogramBackprojection;
using object_detection::Detection;
using object_detection::TrainingData;
using object_detection::histogram_utilities::calculateHistogram;
using object_detection::histogram_utilities::calculateBackprojection;
using object_detection::histogram_utilities::showHSHistogram;
using object_detection::histogram_utilities::calculateHistogram;

static const int NUM_HUE_BINS = 30;
static const int NUM_SATURATION_BINS = 32;
static const double BACKPROJECTION_THRESHOLD = 10.0 / 255.0;

// minimum number of pixels that form an object contour.
// smaller objects are discarded
static const double CONTOUR_AREA_THRESHOLD = 20;


HistogramBackprojection::HistogramBackprojection() : is_trained_(false)
{
}


void HistogramBackprojection::train(const TrainingData& training_data)
{
    // check input
    if (!training_data.isValid())
    {
        throw std::runtime_error("HistogramBackprojection::train(): input data invalid");
    }
    // create object mask
    cv::Mat object_mask = cv::Mat::zeros(training_data.image.rows, 
            training_data.image.cols, CV_8UC1);
    paintFilledPolygon(object_mask, training_data.object_outline,
            cv::Scalar(255));

    // convert image to hsv
    cv::Mat hsv_image;
    cv::cvtColor(training_data.image, hsv_image, CV_BGR2HSV);

    cv::MatND object_histogram = calculateHistogram(hsv_image, NUM_HUE_BINS,
            NUM_SATURATION_BINS, object_mask);

    cv::Mat background_mask = 255 - object_mask;
    cv::MatND background_histogram = calculateHistogram(hsv_image, NUM_HUE_BINS,
            NUM_SATURATION_BINS, background_mask);

    // divide the histograms so that only those colors remain
    // that are representative for the object
    object_histogram_ = object_histogram / (background_histogram + 1);
    // adjust range
    double max;
    cv::minMaxLoc(object_histogram_, NULL, &max);
    object_histogram_ = object_histogram / max;

    showHSHistogram(object_histogram_, "Significant Object Colors");

    cv::Mat hsv_training_image;
    cv::cvtColor(training_data.image, hsv_training_image, CV_BGR2HSV);
    cv::Mat self_back_projection = calculateBackprojection(object_histogram_,
            hsv_training_image);
    cv::namedWindow("Self backprojection");
    cv::imshow("Self backprojection", self_back_projection * 255);
    
    // TODO compute scoring


    is_trained_ = true;
}

std::vector<cv::Rect> HistogramBackprojection::computeRegionsOfInterest(const cv::Mat& image,
        const std::vector<cv::Rect>& input_rois)
{
    showHSHistogram(image, NUM_HUE_BINS, NUM_SATURATION_BINS, cv::Mat(),
            "Image Histogram");

    if (is_trained_)
    {
        // mask out anything outside input ROIs
        cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
        if (input_rois.size() > 0)
        {
            for (size_t i = 0; i < input_rois.size(); ++i)
            {
                cv::rectangle(mask, input_rois[i], cv::Scalar(255), CV_FILLED);
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
        cv::Mat back_projection = calculateBackprojection(object_histogram_, hsv_image);

        // some closing
        int element_size = 5;
        cv::Mat element = cv::Mat::zeros(element_size, element_size, CV_8UC1);
        cv::circle(element, cv::Point(element_size / 2, element_size / 2), element_size / 2, cv::Scalar(255), -1);
        cv::morphologyEx(back_projection, back_projection, cv::MORPH_CLOSE, element);

        // create threshold
        cv::Mat thresholded;
        cv::threshold(back_projection, thresholded, BACKPROJECTION_THRESHOLD, 255, CV_THRESH_BINARY);
        cv::namedWindow( "Backprojection-thresholded-closed", 1 );
        cv::imshow( "Backprojection-thresholded-closed", thresholded );
     
        // extract contours and discard small elements
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(thresholded, contours, CV_RETR_EXTERNAL, 
                CV_CHAIN_APPROX_SIMPLE);


        std::vector<cv::Rect> rois;
        cv::Mat contour_image = cv::Mat::zeros(thresholded.rows, thresholded.cols, CV_8UC1);
        for (size_t i = 0; i < contours.size(); ++i)
        {
            cv::Mat contour = cv::Mat(contours[i]);
            double area = cv::contourArea(contour);
            if (area > CONTOUR_AREA_THRESHOLD)
            {
                rois.push_back(cv::boundingRect(cv::Mat(contours[i])));
                cv::drawContours(contour_image, contours, i, cv::Scalar(255), CV_FILLED);
            }
        }

        cv::namedWindow( "Backprojection", 1 );
        cv::imshow( "Backprojection", back_projection * 255);
        
        cv::namedWindow( "Contour image" );
        cv::imshow("Contour image", contour_image);
       
        return rois;
    }
    else
    {
        throw std::runtime_error("HistogramBackprojection::computeRegionsOfInterest() called without having trained before");
    }
}

