#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>

#include "surf_extractor.h"

using namespace object_detection;

TEST(SurfExtractor, runTest)
{
    cv::Mat image(600, 800, CV_8UC1, cv::Scalar(0));
    int rect_width = 100;
    int rect_height = 80;
    cv::Point rect_tl(image.cols / 2 - rect_width / 2, image.rows / 2 - rect_height / 2);
    cv::Point rect_br = rect_tl + cv::Point(rect_width, rect_height);
    cv::rectangle(image, rect_tl, rect_br, cv::Scalar::all(255), CV_FILLED);

    cv::Point shift(0, 0);

    cv::Mat image2(600, 800, CV_8UC1, cv::Scalar(0));
    cv::rectangle(image2, rect_tl + shift, rect_br + shift, cv::Scalar::all(255), CV_FILLED);
    
    SurfExtractor extractor;
    std::vector<Feature> features = extractor.extract(image);
    SurfExtractor extractor2;
    std::vector<Feature> features2 = extractor2.extract(image2);

    ASSERT_EQ(features.size(), features2.size());
    std::cout << "Surf extracted " << features.size() << " features." << std::endl;

    for (size_t i = 0; i < features.size(); ++i)
    {
        const cv::KeyPoint& kp1 = features[i].key_point;
        const cv::KeyPoint& kp2 = features2[i].key_point;
        const std::vector<float>& desc1 = features[i].descriptor;
        const std::vector<float>& desc2 = features2[i].descriptor;
        EXPECT_EQ(kp2.pt.x - kp1.pt.x, shift.x);
        EXPECT_EQ(kp2.pt.y - kp1.pt.y, shift.y);
        ASSERT_EQ(desc1.size(), desc2.size());
        for (size_t j = 0; j < desc1.size(); ++j)
        {
            EXPECT_EQ(desc1[i], desc2[i]);
        }
    }

}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

