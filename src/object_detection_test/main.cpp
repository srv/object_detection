#include <iostream>

#include <cv.h>
#include <highgui.h>

#include "histogram_backprojection.h"
#include "detection.h"
#include "training_data.h"
#include "utilities.h"

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "ERROR: no image file specified!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <training image file name> <object x> <object y> <object width> <object height> <object angle> <test image file name>" << std::endl;
        std::cerr << "  where <object angle> is given in degrees" << std::endl;
        return -1;
    }
    const char* training_image_name = argv[1];
    int object_x = atoi(argv[2]);
    int object_y = atoi(argv[3]);
    int object_width = atoi(argv[4]);
    int object_height = atoi(argv[5]);
    int object_angle = atoi(argv[6]);

    const char* test_image_name = argv[7];

    std::cout << "Training image: " << training_image_name << std::endl;
    std::cout << "Test image: " << test_image_name << std::endl;
    std::cout << "Object center: " << object_x << ", " << object_y << std::endl;
    std::cout << "Object dimensions: " << object_width << " x "
        << object_height << std::endl;
    std::cout << "Object angle: " << object_angle << std::endl;


    // read and display images
    cv::Mat training_image = cv::imread(training_image_name);
    cv::namedWindow("Training image");
    cv::imshow("Training image", training_image);

    cv::Mat test_image = cv::imread(test_image_name);
    cv::namedWindow("Test image");
    cv::imshow("Test image", test_image);

    // prepare traing data
    object_detection::TrainingData training_data;
    training_data.image = training_image;
    cv::Point2f     object_center(object_x, object_y);
    cv::Size        object_size(object_width, object_height);
    cv::RotatedRect object_pose(object_center, object_size, object_angle);
    training_data.bounding_rotated_rect = object_pose;
 
    cv::Mat training_image_with_marked_object = training_image.clone();
    object_detection::paintRotatedRectangle(training_image_with_marked_object,
            object_pose, cv::Scalar(0, 255, 0));
    cv::namedWindow("Training image with marked object");
    cv::imshow("Training image with marked object", training_image_with_marked_object);

    

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

    
    cv::waitKey();

    return 0;
}

