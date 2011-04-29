
#include <stdexcept>

#include <highgui.h>

#include "detector.h"
#include "detection.h"
#include "object_parts_detector.h"
#include "colored_parts_classifier.h"
#include "training_data.h"
#include "utilities.h"
#include "statistics.h"

using object_detection::Detector;
using object_detection::Detection;
using object_detection::paintFilledPolygon;


Detector::Detector() : is_trained_(false)
{
    boost::shared_ptr<PartsClassifier> colored_parts_classifier = 
        boost::shared_ptr<PartsClassifier>(new ColoredPartsClassifier());

    boost::shared_ptr<ObjectPartsDetector> color_parts_detector =
        boost::shared_ptr<ObjectPartsDetector>(new ObjectPartsDetector(colored_parts_classifier));

    object_parts_detectors_.push_back(color_parts_detector);
}


void Detector::train(const TrainingData& training_data)
{
    // check input
    if (!training_data.isValid())
    {
        throw std::runtime_error("Detector::train(): input data invalid");
    }

    // create object mask
    cv::Mat object_mask = cv::Mat::zeros(training_data.image.rows, 
            training_data.image.cols, CV_8UC1);
    paintFilledPolygon(object_mask, training_data.object_outline,
            cv::Scalar(255));

    for(size_t i = 0; i < object_parts_detectors_.size(); ++i)
    {
        object_parts_detectors_[i]->train(training_data.image, object_mask);
    }

    cv::Mat outline_points_matrix = cv::Mat(training_data.object_outline);
    cv::Scalar centroid = cv::mean(outline_points_matrix);
     
    cv::Mat centered_outline_points_matrix = outline_points_matrix - centroid;
    centered_object_outline_ = cv::Mat_<cv::Point>(centered_outline_points_matrix);

    is_trained_ = true;
}

std::vector<Detection> Detector::detect(const cv::Mat& image,
        const std::vector<cv::Rect>& rois)
{
    if (!is_trained_)
    {
        throw std::runtime_error("Error: Detector::detect() called without having trained before!");
    }

    std::vector<Detection> all_detections;

    for(size_t i = 0; i < object_parts_detectors_.size(); ++i)
    {
        std::vector<Detection> detections = object_parts_detectors_[i]->detect(image);

        // compute outline for detections as parts detector do not know
        // about outlines
        // TODO change this!
        for (size_t i = 0; i < detections.size(); ++i)
        {
            Detection& detection = detections[i]; 
            cv::Mat rotation_matrix = cv::getRotationMatrix2D(cv::Point2f(0.0, 0.0),
                    -detection.angle / M_PI * 180.0, detection.scale);
            cv::Mat rotated_points_matrix;
            cv::transform(cv::Mat(centered_object_outline_), rotated_points_matrix, 
                    rotation_matrix);
            cv::add(rotated_points_matrix, 
                    cv::Scalar(detection.center.x, detection.center.y), 
                    rotated_points_matrix);
            detection.outline = cv::Mat_<cv::Point>(rotated_points_matrix);
            cv::Scalar new_center = cv::mean(cv::Mat(detection.outline));
            detection.center.x = new_center[0];
            detection.center.y = new_center[1];
        }
        all_detections.insert(all_detections.end(), detections.begin(), detections.end());
    }
    return all_detections;
}


