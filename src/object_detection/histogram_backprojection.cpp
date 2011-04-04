
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
static const int BACKPROJECTION_THRESHOLD = 10;

// minimum number of pixels that form an object contour.
// smaller objects are discarded
static const double CONTOUR_AREA_THRESHOLD = 200;


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


    cv::namedWindow("Mask");
    cv::imshow("Mask", object_mask);

    // convert image to hsv
    cv::Mat hsv_image;
    cv::cvtColor(training_data.image, hsv_image, CV_BGR2HSV);

    cv::MatND object_histogram = calculateHistogram(hsv_image, NUM_HUE_BINS,
            NUM_SATURATION_BINS, object_mask);

    cv::Mat background_mask = 255 - object_mask;
    cv::MatND background_histogram = calculateHistogram(hsv_image, NUM_HUE_BINS,
            NUM_SATURATION_BINS, background_mask);

    object_histogram_ = object_histogram / (background_histogram + 1);


//    saturation_threshold_ = object_mean[1] - object_stddev[1];
//    value_minimum_ = object_mean[2] - object_stddev[2];
//    value_maximum_ = object_mean[2] + object_stddev[2];


//    cv::split
//    cv::Mat limit = hsv_image

    showHSHistogram(object_histogram_, "Object histogram (divided)");

    cv::namedWindow("Training image");
    cv::imshow("Training image", training_data.image);

    cv::Mat hsv_training_image;
    cv::cvtColor(training_data.image, hsv_training_image, CV_BGR2HSV);
    cv::Mat self_back_projection = calculateBackprojection(object_histogram_,
            hsv_training_image);
    cv::namedWindow("Self backprojection");
    cv::imshow("Self backprojection", self_back_projection);
    
    // TODO compute scoring


    is_trained_ = true;
}

std::vector<cv::Rect> HistogramBackprojection::computeRegionsOfInterest(const cv::Mat& image,
        const std::vector<cv::Rect>& rois)
{
    showHSHistogram(image, NUM_HUE_BINS, NUM_SATURATION_BINS, cv::Mat(),
            "Image Histogram");

    if (is_trained_)
    {

        //TODO use input ROIs!
    
        // convert input image to hsv
        cv::Mat hsv_image;
        cv::cvtColor(image, hsv_image, CV_BGR2HSV);

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


        // cv::CamShift finds the minimum enclosing rotated bounding rectangle
        /*
        cv::TermCriteria termination_criteria(
                cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
                100,
                3);

        cv::Rect roi(0, 0, image.cols, image.rows);
        cv::RotatedRect object_rect = cv::CamShift(contour_image, roi,
               termination_criteria);


        std::vector<Detection> detections;
        if (object_rect.size.width > 0 && object_rect.size.height > 0)
        {

            Detection detection;
            detection.label = "object";
            detection.bounding_rotated_rect = object_rect;
            detection.score = 0.1;
            detections.push_back(detection);
        }
        */

        cv::namedWindow( "Backprojection", 1 );
        cv::imshow( "Backprojection", back_projection );
        
        cv::namedWindow( "Contour image" );
        cv::imshow("Contour image", contour_image);
       
        return rois;
    }
    else
    {
        throw std::runtime_error("HistogramBackprojection::computeRegionsOfInterest() called without having trained before");
    }
}

