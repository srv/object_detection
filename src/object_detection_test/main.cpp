#include <iostream>
#include <fstream>

#include <cv.h>
#include <highgui.h>

#include "histogram_backprojection.h"
#include "detection.h"
#include "training_data.h"
#include "utilities.h"
#include "detector.h"

using object_detection::Detection;
using object_detection::StereoFeature;

std::vector<cv::Point> readPolygonData(std::istream& in)
{
    std::vector<cv::Point> points;
    while (in.good())
    {
        int x, y;
        in >> x;
        in >> y;
        points.push_back(cv::Point(x, y));
    }
    return points;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "ERROR: wrong number of arguments!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <training image file name> <polygon points file> <test image file name>" << std::endl;
        return -1;
    }
    const char* training_image_name = argv[1];
    const char* polygon_points_file = argv[2];
    const char* test_image_name = argv[3];

    std::cout << "Training image: " << training_image_name << std::endl;
    std::cout << "Polygon points file: " << polygon_points_file << std::endl;
    std::cout << "Test image: " << test_image_name << std::endl;

    // read polygon data
    std::ifstream in(polygon_points_file);
    if (!in.is_open())
    {
        std::cerr << "Error opening poygon points file!" << std::endl;
        return -2;
    }

    std::vector<cv::Point> object_outline = readPolygonData(in);
    std::cout << "read points: " << std::endl;
    for (size_t i = 0; i < object_outline.size(); ++i)
    {
        std::cout << " (" << object_outline[i].x << "," << object_outline[i].y << ")" << std::endl;
    }

    // read and display images
    cv::Mat training_image = cv::imread(training_image_name);
    cv::Mat training_image_with_marked_object = training_image.clone();
    object_detection::paintPolygon(training_image_with_marked_object,
            object_outline, cv::Scalar(0, 255, 0), 2);
    cv::namedWindow("Training image with marked object");
    cv::imshow("Training image with marked object", training_image_with_marked_object);


    cv::Mat test_image = cv::imread(test_image_name);
    cv::namedWindow("Test image");
    cv::imshow("Test image", test_image);

    // prepare traing data
    object_detection::TrainingData training_data;
    training_data.image = training_image;
    training_data.object_outline = object_outline;
    
/*
    object_detection::HistogramBackprojection interest_operator;
    interest_operator.train(training_data);
    
    std::vector<cv::Rect> rois = interest_operator.computeRegionsOfInterest(test_image);

    for(std::vector<cv::Rect>::const_iterator iter = rois.begin();
            iter != rois.end(); ++iter)
    {
        cv::rectangle(test_image, *iter, cv::Scalar(255, 255, 0, 0));
    }

    cv::namedWindow("ROIs", CV_WINDOW_AUTOSIZE);
    cv::imshow("ROIs", test_image);
    */

    // detector interface
    object_detection::Detector detector("detector.cfg");
    detector.train(training_data);

    std::vector<StereoFeature> stereo_features;
    std::vector<Detection> detections = detector.detect(test_image, stereo_features);

    std::cout << "Detector made " << detections.size() << " detections." << std::endl;

    // paint the results
    for (size_t i = 0; i < detections.size(); ++i)
    {
        std::cout << "Detection " << i << ":\n" << detections[i] << std::endl;
        cv::circle(test_image, detections[i].center, 100 * detections[i].scale,
                cv::Scalar(0, 255, 0), 2);
    }
    
    cv::namedWindow("Detections");
    cv::imshow("Detections", test_image);
   
    cv::waitKey();
    return 0;
}

