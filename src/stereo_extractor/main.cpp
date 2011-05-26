#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "stereo_keypoint_extractor.h"

using namespace object_detection;

int main(int argc, char **argv){

    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <left_image> <right_image>" << std::endl;
        std::cout << "  images have to be rectified (points are on same y value in left and right image)" << std::endl;
        return -1;
    }
    cv::Mat image_left = cv::imread(argv[1]);
    cv::Mat image_right = cv::imread(argv[2]);
    cv::namedWindow("left image", CV_WINDOW_NORMAL);
    cv::namedWindow("right image", CV_WINDOW_NORMAL);
    cv::imshow("left image", image_left);
    cv::imshow("right image", image_right);

    StereoKeypointExtractor extractor;
    std::cout << "Extracting and matching keypoints..." << std::flush;
    std::vector<StereoDescriptor> stereo_descriptors = extractor.extract(image_left, image_right);
    std::cout << "found " << stereo_descriptors.size() << " stereo descriptors." << std::endl;

    cv::Mat result_image;
    paintStereoDescriptorMatchings(result_image, image_left, image_right, stereo_descriptors);

    cv::namedWindow("matchings", CV_WINDOW_NORMAL);
    cv::imshow("matchings", result_image);
    cvWaitKey();


    return 0;
}

