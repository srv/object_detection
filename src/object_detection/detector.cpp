#include <highgui.h>

#include "detector.h"
#include "detection.h"
#include "color_classifier.h"
#include "training_data.h"
#include "utilities.h"
#include "statistics.h"

using object_detection::Detector;
using object_detection::Detection;
using boost::shared_ptr;
using object_detection::paintFilledPolygon;
using object_detection::computeStatistics;
using object_detection::computeGaussIntersections;

Detector::Detector() : is_trained_(false)
{
    ClassifierWithInfo color_classifier;
    color_classifier.classifier = shared_ptr<Classifier>(new ColorClassifier());
    classifiers_with_info_.push_back(color_classifier);
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

    for(size_t i = 0; i < classifiers_with_info_.size(); ++i)
    {
        classifiers_with_info_[i].classifier->train(training_data.image, object_mask);
        classifiers_with_info_[i].object_outline = training_data.object_outline;
        cv::Mat classification_result = classifiers_with_info_[i].classifier->classify(training_data.image);
        Statistics object_statistics = computeStatistics(classification_result, object_mask);
        std::cout << "Object statistics: " << object_statistics << std::endl;
        Statistics background_statistics = computeStatistics(classification_result, 255 - object_mask);
        std::cout << "Background statistics: " << background_statistics << std::endl;
        // calculate optimal threshold as intersection of gauss graphs
        std::vector<double> gauss_intersections = computeGaussIntersections(
            object_statistics.mean, object_statistics.stddev,
            background_statistics.mean, background_statistics.stddev);
        if(gauss_intersections.size() > 0)
        {
            classifiers_with_info_[i].threshold = *(std::max_element(gauss_intersections.begin(), gauss_intersections.end()));
            cv::Mat binary;
            cv::threshold(classification_result, binary, classifiers_with_info_[i].threshold, 1.0, CV_THRESH_BINARY);
            imshow(classifiers_with_info_[i].classifier->getName() + " thresholded", binary);
            classifiers_with_info_[i].object_statistics = computeStatistics(binary, object_mask);
        }
    }

    is_trained_ = true;

}

std::vector<Detection> Detector::detect(const cv::Mat& image,
        const std::vector<cv::Rect>& rois)
{
    if (!is_trained_)
    {
        throw std::runtime_error("Error: Detector::detect() called without having trained before!");
    }

    std::vector<Detection> detections;

    std::vector<cv::Mat> classification_results(classifiers_with_info_.size());
    for(size_t i = 0; i < classifiers_with_info_.size(); ++i)
    {
        classification_results[i] = classifiers_with_info_[i].classifier->classify(image, rois);
        cv::imshow(classifiers_with_info_[i].classifier->getName(), classification_results[i]);
        cv::Mat binary;
        cv::threshold(classification_results[i], binary, classifiers_with_info_[i].threshold, 1.0, CV_THRESH_BINARY);
        imshow(classifiers_with_info_[i].classifier->getName() + " thresholded", binary);
        Statistics detected_object_statistics = computeStatistics(binary);
        Detection detection;
        detection.angle = detected_object_statistics.main_axis_angle;
        detection.center = detected_object_statistics.center_of_mass;
        detection.scale = detected_object_statistics.area / classifiers_with_info_[i].object_statistics.area;
        detection.score = 0.0;
        detection.label = "object1-" + classifiers_with_info_[i].classifier->getName();

        // rotate polygon outline
        cv::Mat points = cv::Mat(classifiers_with_info_[i].object_outline);
        std::cout << "points: " << points << std::endl;
        cv::Mat centered_points;
        cv::subtract(points, 
                cv::Scalar(classifiers_with_info_[i].object_statistics.center_of_mass.x,
                           classifiers_with_info_[i].object_statistics.center_of_mass.y), centered_points); 
        std::cout << "centered points: " << centered_points << std::endl;
        cv::Mat rotation_matrix =
            cv::getRotationMatrix2D(cv::Point2f(0.0, 0.0), -detection.angle / M_PI * 180.0, detection.scale);
        cv::Mat rotated_points_matrix;
        cv::transform(centered_points, rotated_points_matrix, rotation_matrix);
        cv::add(rotated_points_matrix, cv::Scalar(detection.center.x, detection.center.y), rotated_points_matrix);
        detection.outline = cv::Mat_<cv::Point>(rotated_points_matrix);

        detections.push_back(detection);
    }
    return detections;
}

