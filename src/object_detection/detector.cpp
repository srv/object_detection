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
    classifiers_.push_back(shared_ptr<Classifier>(new ColorClassifier()));
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

    for(size_t i = 0; i < classifiers_.size(); ++i)
    {
        classifiers_[i]->train(training_data.image, object_mask);
        cv::Mat classification_result = classifiers_[i]->classify(training_data.image);
        object_statistics_ = computeStatistics(classification_result, object_mask);
        std::cout << "Object statistics: " << object_statistics_ << std::endl;
        background_statistics_ = computeStatistics(classification_result, 255 - object_mask);
        std::cout << "Background statistics: " << background_statistics_ << std::endl;
        // calculate optimal threshold as intersection of gauss graphs
        std::vector<double> gauss_intersections = computeGaussIntersections(
            object_statistics_.mean, object_statistics_.stddev,
            background_statistics_.mean, background_statistics_.stddev);
        if(gauss_intersections.size() > 0)
        {
            double thresh = *(std::max_element(gauss_intersections.begin(), gauss_intersections.end()));
            cv::Mat binary;
            cv::threshold(classification_result, binary, thresh, 1.0, CV_THRESH_BINARY);
            imshow(classifiers_[i]->getName() + " thresholded", binary);
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

    std::vector<cv::Mat> classification_results(classifiers_.size());
    for(size_t i = 0; i < classifiers_.size(); ++i)
    {
        classification_results[i] = classifiers_[i]->classify(image, rois);
        cv::imshow(classifiers_[i]->getName(), classification_results[i]);
    }

    return detections;
}



