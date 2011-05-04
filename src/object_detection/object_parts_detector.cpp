
#include <stdexcept>
#include <cv.h>
#include <highgui.h>

#include "object_parts_detector.h"
#include "parts_classifier.h"
#include "detection.h"
#include "statistics.h"
#include "shape_matching.h"

using object_detection::ObjectPartsDetector;
using object_detection::PartsClassifier;
using object_detection::Detection;
using object_detection::Statistics;

static const double MIN_SCALE = 0.2;
static const double MAX_SCALE = 5.0;

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

    // apply directly to training image to get object part description
    cv::Mat prob_image = parts_classifier_->classify(image);
    threshold_ = computeBestThreshold(prob_image, object_mask);

    cv::Mat thresholded;
    cv::threshold(prob_image, thresholded, threshold_, 1.0, CV_THRESH_BINARY);

    // store shapes as object description
    cv::Mat masked_prob_image;
    thresholded.copyTo(masked_prob_image, object_mask);
    object_part_shapes_ = extractShapes(masked_prob_image);

    cv::Mat object_shapes_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::drawContours(object_shapes_image, object_part_shapes_, -1, cv::Scalar(255), CV_FILLED);

    cv::imshow(parts_classifier_->getName() + " object shapes", object_shapes_image);

    object_part_statistics_ = computeStatistics(object_shapes_image);

    Statistics full_object_statistics = computeStatistics(object_mask);

    relative_object_center_ = full_object_statistics.center_of_mass -
        object_part_statistics_.center_of_mass;
}

std::vector<Detection> ObjectPartsDetector::detect(const cv::Mat& image)
{
    assert(!image.empty());
    std::vector<Detection> detections;
    if (object_part_shapes_.size() == 0)
    {
        return detections;
    }

    cv::Mat prob_image = parts_classifier_->classify(image);
    cv::Mat prob_image_thresholded;
    cv::threshold(prob_image, prob_image_thresholded, threshold_, 1.0, CV_THRESH_BINARY);
    std::vector<std::vector<cv::Point> > detected_shapes = extractShapes(prob_image_thresholded);
    cv::Mat shapes_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::drawContours(shapes_image, detected_shapes, -1, cv::Scalar(255), CV_FILLED);
    cv::imshow(parts_classifier_->getName() + " detected shapes", shapes_image);

    // did we detect some shapes?
    if (detected_shapes.size() > 0)
    {
        double score;
        ShapeMatching::MatchingParameters match_parameters =
            ShapeMatching::matchShapes(object_part_shapes_[0],
                    detected_shapes[0], &score);
        Statistics detected_object_statistics = computeStatistics(shapes_image);
        // check if detection is plausible
        if (match_parameters.scale >= MIN_SCALE && match_parameters.scale <= MAX_SCALE)
        {
            //double distance = cv::matchShapes(cv::Mat(object_part_shapes_[0]), cv::Mat(detected_shapes[0]), CV_CONTOURS_MATCH_I1, 0.0);
            //double score = exp(-distance);
            Detection detection;
            //detection.angle = detected_object_statistics.main_axis_angle - object_part_statistics_.main_axis_angle;
            detection.angle = match_parameters.rotation;
            // compute center
            cv::Point center;
            double sinAngle = sin(-match_parameters.rotation);
            double cosAngle = cos(-match_parameters.rotation);
            center.x = cosAngle * relative_object_center_.x + sinAngle * relative_object_center_.y;
            center.y = -sinAngle * relative_object_center_.x + cosAngle * relative_object_center_.y;
            center.x *= match_parameters.scale;
            center.y *= match_parameters.scale;
            center.x += detected_object_statistics.center_of_mass.x;
            center.y += detected_object_statistics.center_of_mass.y;
            detection.center = center;
            detection.scale = match_parameters.scale;
            detection.score = score;
            detection.label = "object1";
            //outline is added in detector
            //detection.outline = detected_shapes[0];
            detections.push_back(detection);
        }
    }
    return detections;
}

double ObjectPartsDetector::computeBestThreshold(const cv::Mat& image,
        const cv::Mat& mask)
{
    Statistics object_statistics = computeStatistics(image, mask);
    Statistics background_statistics = computeStatistics(image, 
            cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255)) - mask);

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
    return threshold;
}

std::vector<std::vector<cv::Point> > 
ObjectPartsDetector::extractShapes(const cv::Mat& image)
{
    cv::Mat image_copy;
    image.convertTo(image_copy, CV_8UC1);
    std::vector<std::vector<cv::Point> > contours;
    //cv::findContours(image_copy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::findContours(image_copy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);
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
