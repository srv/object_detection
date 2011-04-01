#include <iostream>

#include <cv.h>
#include <highgui.h>

#include "detection.h"
#include "training_data.h"
#include "utilities.h"
#include "histogram_utilities.h"

namespace hu = object_detection::histogram_utilities;

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        std::cerr << "ERROR: no image file specified!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <image file name> <object x> <object y> <object width> <object height> <object angle>" << std::endl;
        std::cerr << "  where <object angle> is given in degrees" << std::endl;
        return -1;
    }
    const char* image_name = argv[1];
    int object_x = atoi(argv[2]);
    int object_y = atoi(argv[3]);
    int object_width = atoi(argv[4]);
    int object_height = atoi(argv[5]);
    int object_angle = atoi(argv[6]);

    std::cout << "Image: " << image_name << std::endl;
    std::cout << "Object center: " << object_x << ", " << object_y << std::endl;
    std::cout << "Object dimensions: " << object_width << " x "
        << object_height << std::endl;
    std::cout << "Object angle: " << object_angle << std::endl;

    // read image
    cv::Mat image = cv::imread(image_name);

    // prepare rect data
    cv::Point2f     object_center(object_x, object_y);
    cv::Size        object_size(object_width, object_height);
    cv::RotatedRect object_pose(object_center, object_size, object_angle);
 
    cv::Mat image_with_marked_object = image.clone();
    object_detection::paintRotatedRectangle(image_with_marked_object,
            object_pose, cv::Scalar(0, 255, 0), 3);
    cv::namedWindow("Image with marked object");
    cv::imshow("Image with marked object", image_with_marked_object);

    // create object and background masks
    cv::Mat object_mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    object_detection::paintFilledRotatedRectangle(object_mask, object_pose, cv::Scalar(255));
    cv::Mat background_mask = 255 - object_mask;

    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, CV_BGR2HSV);

    cv::MatND object_histogram = hu::calculateHistogram(hsv_image, 32, 32, object_mask);
    cv::MatND background_histogram = hu::calculateHistogram(hsv_image, 32, 32, background_mask);
    hu::showHSHistogram(object_histogram, "object histogram");
    hu::showHSHistogram(background_histogram, "background histogram");

    cv::MatND division_histogram = object_histogram / (background_histogram + 1);
    hu::showHSHistogram(division_histogram, "division histogram");

    cv::Mat object_histogram_backprojection = hu::calculateBackprojection(object_histogram, hsv_image);
    cv::Mat division_histogram_backprojection = hu::calculateBackprojection(division_histogram, hsv_image);

    cv::namedWindow("object histogram backprojection");
    cv::imshow("object histogram backprojection", object_histogram_backprojection);

    cv::namedWindow("division histogram backprojection");
    cv::imshow("division histogram backprojection", division_histogram_backprojection);
    cv::waitKey();

    return 0;
}

