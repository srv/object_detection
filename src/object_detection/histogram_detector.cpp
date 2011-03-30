
#include <iostream>
#include <stdexcept>
#include <highgui.h>

#include "histogram_detector.h" 
#include "detection.h"
#include "training_data.h"
#include "utilities.h"


using namespace object_detection;

static const int NUM_HUE_BINS = 30;
static const int NUM_SATURATION_BINS = 32;

// minimum number of pixels that form an object contour.
// smaller objects are discarded
static const double CONTOUR_AREA_THRESHOLD = 500;


HistogramDetector::HistogramDetector() : is_trained_(false)
{
}


void HistogramDetector::train(const TrainingData& training_data)
{
    // check input
    if (!training_data.isValid())
    {
        throw std::runtime_error("HistogramDetector::train(): input data invalid");
    }
    // create object mask
    cv::Mat object_mask = cv::Mat::zeros(training_data.image.rows, 
            training_data.image.cols, CV_8UC1);
    paintFilledRotatedRectangle(object_mask, training_data.bounding_rotated_rect,
            cv::Scalar(255));

    cv::namedWindow("Mask");
    cv::imshow("Mask", object_mask);

    // convert image to hsv
    cv::Mat hsv_image;
    cv::cvtColor(training_data.image, hsv_image, CV_BGR2HSV);

    object_histogram_ = calculateHistogram(hsv_image, NUM_HUE_BINS,
            NUM_SATURATION_BINS, object_mask);

    // calculate some statistics
    cv::Scalar mean, stddev;
    cv::meanStdDev(hsv_image, mean, stddev, object_mask);
    std::cout << "object statistics: " << std::endl;
    std::cout << "     mean = " << mean[0] << "," << mean[1] << "," << mean[2] << std::endl;
    std::cout << "   stddev = " << stddev[0] << "," << stddev[1] << "," << stddev[2] << std::endl;

    cv::Mat background_mask = 255 - object_mask;
    cv::meanStdDev(hsv_image, mean, stddev, background_mask);
    std::cout << "background statistics: " << std::endl;
    std::cout << "     mean = " << mean[0] << "," << mean[1] << "," << mean[2] << std::endl;
    std::cout << "   stddev = " << stddev[0] << "," << stddev[1] << "," << stddev[2] << std::endl;

    showHSHistogram(object_histogram_, "Object histogram");

    cv::namedWindow("Training image");
    cv::imshow("Training image", training_data.image);

    cv::Mat hsv_training_image;
    cv::cvtColor(training_data.image, hsv_training_image, CV_BGR2HSV);
    cv::Mat self_back_projection = calculateBackprojection(object_histogram_,
            hsv_training_image);
    cv::namedWindow("Self backprojection");
    cv::imshow("Self backprojection", self_back_projection);
    
    // TODO compute scoring


    object_size_ = training_data.bounding_rotated_rect.size;
    
    is_trained_ = true;
}

std::vector<Detection> HistogramDetector::detect(const cv::Mat& image,
        const std::vector<cv::Rect>& rois)
{
    showHSHistogram(image, NUM_HUE_BINS, NUM_SATURATION_BINS, cv::Mat(),
            "Image Histogram");

    if (is_trained_)
    {
    
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
        cv::threshold(back_projection, thresholded, 100, 255, CV_THRESH_BINARY);
        cv::namedWindow( "Backprojection-thresholded-closed", 1 );
        cv::imshow( "Backprojection-thresholded-closed", thresholded );
     
        // extract contours and discard small elements
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(thresholded, contours, CV_RETR_EXTERNAL, 
                CV_CHAIN_APPROX_SIMPLE);


        cv::Mat contour_image = cv::Mat::zeros(thresholded.rows, thresholded.cols, CV_8UC1);
        for (size_t i = 0; i < contours.size(); ++i)
        {
            cv::Mat contour = cv::Mat(contours[i]);
            double area = cv::contourArea(contour);
            if (area > CONTOUR_AREA_THRESHOLD)
            {
                cv::drawContours(contour_image, contours, i, cv::Scalar(255), CV_FILLED);
            }
        }


        // cv::CamShift finds the minimum enclosing rotated bounding rectangle
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

        cv::namedWindow( "Backprojection", 1 );
        cv::imshow( "Backprojection", back_projection );
        
        cv::namedWindow( "Contour image" );
        cv::imshow("Contour image", contour_image);
       
        return detections;
    }
    else
    {
        throw std::runtime_error("HistogramDetector::detect() called without having trained before");
    }
}

// calculates a two dimensional hue-saturation histogram
cv::MatND HistogramDetector::calculateHistogram(const cv::Mat& hsv_image,
        int num_hue_bins, int num_saturation_bins, const cv::Mat& mask)
{   
    // we assume that the image is a regular
    // three channel image
    CV_Assert(hsv_image.type() == CV_8UC3);

    // dimensions of the histogram
    int histogram_size[] = {num_hue_bins, num_saturation_bins};

    // ranges for the histogram
    float hue_ranges[] = {0, 180};
    float saturation_ranges[] = {0, 256};
    const float* ranges[] = {hue_ranges, saturation_ranges};

    // channels for wich to compute the histogram (H and S)
    int channels[] = {0, 1};

    cv::MatND histogram;

    // calculation
    int num_arrays = 1;
    int dimensions = 2;
    cv::calcHist(&hsv_image, num_arrays, channels, mask, histogram, dimensions,
            histogram_size, ranges);

    return histogram;
}

cv::Mat HistogramDetector::calculateBackprojection(const cv::MatND& histogram,
        const cv::Mat& hsv_image)
{
    // we assume that the image is a regular
    // three channel image
    CV_Assert(hsv_image.type() == CV_8UC3);

    // channels for wich to compute the histogram (H and S)
    int channels[] = {0, 1};

    // ranges for the histogram
    float hue_ranges[] = {0, 180};
    float saturation_ranges[] = {0, 256};
    const float* ranges[] = {hue_ranges, saturation_ranges};

    cv::Mat back_projection;
    int num_arrays = 1;
    cv::calcBackProject(&hsv_image, num_arrays, channels, histogram,
           back_projection, ranges);

    return back_projection;
}


void HistogramDetector::showHSHistogram(const cv::Mat& image,
        int num_hue_bins, int num_saturation_bins, const cv::Mat& mask,
        const std::string& name)
{
    // we assume that the image is a regular
    // three channel image
    CV_Assert(image.type() == CV_8UC3);
    
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, CV_BGR2HSV);

    cv::MatND histogram = calculateHistogram(hsv_image, num_hue_bins,
            num_saturation_bins, mask);
    showHSHistogram(histogram, name);
}
 
void HistogramDetector::showHSHistogram(const cv::MatND& histogram,
        const std::string& name)
{
    int num_hue_bins = histogram.cols;
    int num_saturation_bins = histogram.rows;

    // visualization
    double max_value = 0;
    cv::minMaxLoc(histogram, 0, &max_value, 0, 0);
    int scale = 10;
    cv::Mat histogram_image_hsv = 
        cv::Mat::zeros((num_saturation_bins + 1) * scale, (num_hue_bins + 1) * scale, CV_8UC3);

    // paint x axis
    for( int h = 1; h < num_hue_bins + 1; h++ )
        cv::rectangle( histogram_image_hsv, cv::Point(h*scale, 0),
                cv::Point((h + 1)*scale - 1, scale - 1),
                cv::Scalar(1.0 * (h - 1) / num_hue_bins * 180.0, 255, 255, 0),
                CV_FILLED);

    // paint y axis
    for( int s = 1; s < num_saturation_bins + 1; s++ )
        cv::rectangle( histogram_image_hsv, cv::Point(0, s * scale),
                cv::Point(scale - 1, (s + 1)*scale - 1),
                cv::Scalar(180, 1.0 * (s - 1) / num_saturation_bins * 255.0, 255, 0),
                CV_FILLED);

    cv::Mat histogram_image;
    cv::cvtColor(histogram_image_hsv, histogram_image, CV_HSV2BGR);

    for( int h = 1; h < num_hue_bins + 1; h++ )
        for( int s = 1; s < num_saturation_bins + 1; s++ )
        {
            float binVal = histogram.at<float>(h - 1, s - 1);
            int intensity = cvRound(binVal * 255 / max_value);
            cv::rectangle( histogram_image, cv::Point(h*scale, s*scale),
                cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                cv::Scalar::all(intensity),
                CV_FILLED );
         }

    cv::namedWindow( name );
    cv::imshow( name, histogram_image );
}

