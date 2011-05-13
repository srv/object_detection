#include <iostream>
#include <fstream>
#include <sstream>

#include <cv.h>
#include <highgui.h>

#include "glcm.h"

using object_detection::computeGLCM;
using object_detection::computeGLCMFeatures;


void normalize(cv::Mat& image)
{
    double max;
    cv::minMaxLoc(image, NULL, &max);
    image /= max;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <image file name> <dx> <dy> <size>" << std::endl;
        return -1;
    }
    const char* image_name = argv[1];
    int dx = 1;
    int dy = 0;
    int size = 32;
    if (argc > 2)
    {
        dx = atoi(argv[2]);
    }
    if (argc > 3)
    {
        dy = atoi(argv[3]);
    }
    if (argc > 4)
    {
        size = atoi(argv[4]);
    }

    std::cout << "Image: " << image_name << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "dy = " << dy << std::endl;
    std::cout << "size = " << size << std::endl;

    // read and display images
    cv::Mat image = cv::imread(image_name);
    cv::namedWindow("Input image");
    cv::imshow("Input image", image);

    cv::Mat grayscale_image;
    cv::cvtColor(image, grayscale_image, CV_BGR2GRAY);
    cv::imshow("gray image", grayscale_image);

    // sliding window glcm
    int window_size = 15;
    int border_size = window_size / 2;
    cv::Mat bordered_image;
    cv::copyMakeBorder(grayscale_image, bordered_image, border_size, border_size,
            border_size, border_size, cv::BORDER_REFLECT_101);

    cv::Mat disimilarity(image.rows, image.cols, CV_32F);
    cv::Mat uniformity(image.rows, image.cols, CV_32F);
    cv::Mat entropy(image.rows, image.cols, CV_32F);
    cv::Mat contrast(image.rows, image.cols, CV_32F);

    for(int r = 0; r < image.rows; ++r)
    {
        for(int c = 0; c < image.cols; ++c)
        {
            cv::Rect roi(c, r, window_size, window_size);
            cv::Mat window(bordered_image, roi);
            cv::Mat glcm = computeGLCM(window, dx, dy, size);
            cv::Scalar texture_features = computeGLCMFeatures(glcm);
            disimilarity.at<float>(r, c) = texture_features[0];
            uniformity.at<float>(r, c) = texture_features[1];
            entropy.at<float>(r, c) = texture_features[2];
            contrast.at<float>(r, c) = texture_features[3];
        }
    }

    normalize(disimilarity);
    normalize(uniformity);
    normalize(entropy);
    normalize(contrast);

    cv::imshow("disimilarity", disimilarity);
    cv::imshow("uniformity", uniformity);
    cv::imshow("entropy", entropy);
    cv::imshow("contrast", contrast);

    cv::waitKey();
    return 0;
}

