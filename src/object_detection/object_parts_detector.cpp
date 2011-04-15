
#include <cv.h>
#include <highgui.h>

#include "object_parts_detector.h"
#include "parts_classifier.h"
#include "detection.h"
#include "statistics.h"

using object_detection::ObjectPartsDetector;
using object_detection::PartsClassifier;
using object_detection::Detection;
using object_detection::Statistics;


bool compareShapeArea(const std::vector<cv::Point>& shape1,
                      const std::vector<cv::Point>& shape2)
{
    return cv::contourArea(cv::Mat(shape1)) > cv::contourArea(cv::Mat(shape2));
}

ObjectPartsDetector::ObjectPartsDetector(
        boost::shared_ptr<PartsClassifier> parts_classifier) :
    parts_classifier_(parts_classifier)
{
    if (parts_classifier_.get() == NULL)
    {
        throw std::runtime_error(
            "ObjectPartsDetector::ObjectPartsDetector(): given PartsClassifier pointer is NULL!");
    }

}
	
void ObjectPartsDetector::train(const cv::Mat& image, const cv::Mat& object_mask)
{
    assert(!image.empty());
    assert(!object_mask.empty());
    assert(object_mask.rows == image.rows);
    assert(object_mask.cols == image.cols);

    // train the classifier
    parts_classifier_->train(image, object_mask);

    // apply directly to training image to get object description
    cv::Mat prob_image = parts_classifier_->classify(image);
    threshold_ = computeBestThreshold(prob_image, object_mask);

    cv::Mat thresholded;
    cv::threshold(prob_image, thresholded, threshold_, 1.0, CV_THRESH_BINARY);

    // store shapes as object description
    cv::Mat masked_prob_image;
    thresholded.copyTo(masked_prob_image, object_mask);
    object_shapes_ = extractShapes(masked_prob_image);

    std::cout << "object has " << object_shapes_.size() << " shapes." << std::endl;

    cv::Mat biggestContoursImage = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::drawContours(biggestContoursImage, object_shapes_, -1, cv::Scalar(255));

    cv::imshow(parts_classifier_->getName() + " main object parts", biggestContoursImage);

    object_statistics_ = computeStatistics(biggestContoursImage);
}

std::vector<Detection> ObjectPartsDetector::detect(const cv::Mat& image)
{
    cv::Mat prob_image = parts_classifier_->classify(image);
    cv::imshow(parts_classifier_->getName() + " object parts prob image", prob_image);
    cv::Mat binary;
    cv::threshold(prob_image, binary, threshold_, 1.0, CV_THRESH_BINARY);
    cv::imshow(parts_classifier_->getName() + " object parts prob image thresholded", binary);
    std::vector<std::vector<cv::Point> > shapes = extractShapes(binary);

    cv::Mat biggestPartsImage = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::drawContours(biggestPartsImage, shapes, -1, cv::Scalar(255));
    cv::imshow(parts_classifier_->getName() + " detected object parts", biggestPartsImage);

    std::vector<Detection> detections;
    if (shapes.size() > 0)
    {
        Statistics detected_object_statistics = computeStatistics(biggestPartsImage);
        std::cout << "object statistics: " << object_statistics_ << std::endl;
        std::cout << "detected object statistics: " << detected_object_statistics << std::endl;
        // TODO shape matching etc.
        Detection detection;
        detection.angle = 0.0;
        detection.center = detected_object_statistics.center_of_mass;
        detection.scale = detected_object_statistics.area / object_statistics_.area;
        detection.score = 0.0;
        detection.label = "object1";
        detections.push_back(detection);
    }
    return detections;
}

double ObjectPartsDetector::computeBestThreshold(const cv::Mat& image,
        const cv::Mat& mask)
{
    Statistics object_statistics = computeStatistics(image, mask);
    Statistics background_statistics = computeStatistics(image, 255 - mask);

    std::cout << "object_statistics: " << object_statistics << std::endl;
    std::cout << "background_statistics: " << background_statistics << std::endl;

    // TODO !!
    if (background_statistics.mean < 0.000001)
    {
        return object_statistics.mean / 2;
    }

    std::vector<double> gauss_intersections = computeGaussIntersections(
            object_statistics.mean, object_statistics.stddev,
            background_statistics.mean, background_statistics.stddev);
    if(gauss_intersections.size() == 0)
    {
        throw std::runtime_error("ObjectPartsDetector::computeBestThreshold(): no best threshold computable");
    }
     
    double threshold = *(std::max_element(gauss_intersections.begin(), 
                                          gauss_intersections.end()));

    std::cout << "threshold: " << threshold << std::endl;
    return threshold;
}

std::vector<std::vector<cv::Point> > 
ObjectPartsDetector::extractShapes(const cv::Mat& image)
{
    cv::Mat image_copy;
    image.convertTo(image_copy, CV_8UC1);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(image_copy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    std::cout << contours.size() << " contours" << std::endl;
    return getBiggestShapes(contours);
}

std::vector<std::vector<cv::Point> > ObjectPartsDetector::getBiggestShapes(
        const std::vector<std::vector<cv::Point> >& shapes)
{
    std::vector<std::vector<cv::Point> > sorted_shapes = shapes;
    std::sort(sorted_shapes.begin(), sorted_shapes.end(), compareShapeArea);
    std::vector<std::vector<cv::Point> > biggest_shapes;
    if (sorted_shapes.size() > 0)
    {
        double biggest_area = cv::contourArea(cv::Mat(sorted_shapes[0]));
        std::vector<std::vector<cv::Point> >::const_iterator iter = 
            sorted_shapes.begin();
        bool stop = false;
        while (!stop && iter != sorted_shapes.end())
        {
            if (cv::contourArea(cv::Mat(*iter)) > 0.5 * biggest_area)
            {
                biggest_shapes.push_back(*iter);
            }
            else
            {
                stop = true;
            }
            ++iter;
        }
    }
    return biggest_shapes;
}
