#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>

#include "stereo_feature_extractor.h"

using namespace object_detection;

TEST(StereoFeatureExtractor, runTest)
{
    cv::Mat image_left(240, 320, CV_8UC3);
    cv::Mat image_right(240, 320, CV_8UC3);

    //cv::randu(image_left, cv::Scalar(0, 0, 0), cv::Scalar(50, 50, 50));
    //cv::randu(image_right, cv::Scalar(0, 0, 0), cv::Scalar(50, 50, 50));
    image_left = cv::Scalar::all(0);
    image_right = cv::Scalar::all(0);

    cv::rectangle(image_left, cv::Point(20, 20), cv::Point(120, 120), cv::Scalar::all(255), CV_FILLED);
    cv::rectangle(image_right, cv::Point(21, 21), cv::Point(121, 121), cv::Scalar::all(255), CV_FILLED);

    StereoFeatureExtractor extractor;
    std::vector<StereoFeature> stereo_features = extractor.extract(image_left, image_right);
    std::cout << "Found " << stereo_features.size() << " stereo features." << std::endl;

    cv::Mat result_image;
    paintStereoFeatureMatchings(result_image, image_left, image_right, stereo_features);

    cv::imshow("matchings", result_image);
    cvWaitKey(1000);

}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

