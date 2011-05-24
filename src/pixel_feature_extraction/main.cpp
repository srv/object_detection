#include <iostream>
#include <fstream>

#include <cv.h>
#include <highgui.h>

#include "utilities.h"
#include "glcm.h"

using namespace object_detection;

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
        std::cerr << "Usage: " << argv[0] << " <training image file name> <polygon points file> <output weka data file>" << std::endl;
        return -1;
    }
    const char* training_image_name = argv[1];
    const char* polygon_points_file = argv[2];
    const char* data_file = argv[3];

    std::cout << "Training image: " << training_image_name << std::endl;
    std::cout << "Polygon points file: " << polygon_points_file << std::endl;

    // read polygon data
    std::ifstream in(polygon_points_file);
    if (!in.is_open())
    {
        std::cerr << "Error opening poygon points file!" << std::endl;
        return -2;
    }

    std::vector<cv::Point> object_outline = readPolygonData(in);

    // read and display images
    cv::Mat training_image = cv::imread(training_image_name);
    cv::Mat training_image_with_marked_object = training_image.clone();
    object_detection::paintPolygon(training_image_with_marked_object,
            object_outline, cv::Scalar(0, 255, 0));
    cv::namedWindow("Training image with marked object");
    cv::imshow("Training image with marked object", training_image_with_marked_object);

    cv::Mat mask(training_image.rows, training_image.cols, CV_8UC1);
    mask = cv::Scalar(0);
 
    object_detection::paintFilledPolygon(mask, object_outline, cv::Scalar(255));
    cv::namedWindow("Mask");
    cv::imshow("Mask", mask);


    std::ofstream data(data_file);
    if (!data.is_open())
    {
        std::cerr << "Cannot open " << data_file << " for writing!" << std::endl;
        return -3;
    }

    cv::Mat hsv_image(training_image.rows, training_image.cols, CV_8UC3);
    cv::cvtColor(training_image, hsv_image, CV_BGR2HSV);


    std::cout << "computing glcm..." << std::flush;
    cv::Mat texture_image = computeSlidingWindowUniformGLCM(training_image, 32, 5);

    data << "% image data" << std::endl;
    data << "@relation image_data" << std::endl;
    data << "@attribute b real" << std::endl;
    data << "@attribute g real" << std::endl;
    data << "@attribute r real" << std::endl;
    data << "@attribute h real" << std::endl;
    data << "@attribute s real" << std::endl;
    data << "@attribute v real" << std::endl;
    data << "@attribute disimilarity real" << std::endl;
    data << "@attribute uniformity real" << std::endl;
    data << "@attribute entropy real" << std::endl;
    data << "@attribute glcm_mean real" << std::endl;
    data << "@attribute class {object, background}" << std::endl;
    data << "@data" << std::endl;

    for(int r = 0; r < training_image.rows; ++r)
    {
        for(int c = 0; c < training_image.cols; ++c)
        {
            data << (int)(training_image.at<cv::Vec3b>(r, c)[0]) << ",";
            data << (int)(training_image.at<cv::Vec3b>(r, c)[1]) << ",";
            data << (int)(training_image.at<cv::Vec3b>(r, c)[2]) << ",";
            data << (int)(hsv_image.at<cv::Vec3b>(r, c)[0]) << ",";
            data << (int)(hsv_image.at<cv::Vec3b>(r, c)[1]) << ",";
            data << (int)(hsv_image.at<cv::Vec3b>(r, c)[2]) << ",";
            data << texture_image.at<cv::Vec4f>(r, c)[0] << ",";
            data << texture_image.at<cv::Vec4f>(r, c)[1] << ",";
            data << texture_image.at<cv::Vec4f>(r, c)[2] << ",";
            data << texture_image.at<cv::Vec4f>(r, c)[3] << ",";
            if (mask.at<unsigned char>(r, c) == 0)
            {
                data << "background";
            }
            else
            {
                data << "object";
            }
            data << std::endl;
        }
    }
    std::cout << "data written to " << data_file << std::endl;

    cv::waitKey();
    return 0;
}

