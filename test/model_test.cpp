#include <gtest/gtest.h>

#include "model.h"

using namespace object_detection;

TEST(Model, precondition_assertions)
{
    Model model;
    EXPECT_DEATH(model.getWorldPoint(0), ".*Assert.*");
    
    cv::Point3f point;
    Feature feature;
    model.addFeature(point, feature);
    EXPECT_DEATH(model.getWorldPoint(1),".*Assert.*");
}

TEST(Model, getSet)
{
    Model model;
    for (int i = 0; i < 10; ++i)
    {

        cv::Point3f point(i * 1, i * 2, i * 3);
        Feature feature;
        feature.descriptor.resize(64);
        for (size_t j = 0; j < feature.descriptor.size(); ++j)
        {
            feature.descriptor[j] = i * 100 + (int)j; // cast to match EXPECT_DOUBLE below
        }
        model.addFeature(point, feature);
    }

    cv::Mat feature_data = model.getFeatureData();

    EXPECT_EQ(feature_data.rows, 10);
    EXPECT_EQ(feature_data.cols, 64);
    ASSERT_EQ(feature_data.type(), CV_32F);
    for (int r = 0; r < feature_data.rows; ++r)
    {
        for (int c = 0; c < feature_data.cols; ++c)
        {
            EXPECT_DOUBLE_EQ(feature_data.at<float>(r, c), r * 100 + c);
        }
    }

    for (int i = 0; i < 10; ++i)
    {
        cv::Point3f point = model.getWorldPoint(i);
        EXPECT_DOUBLE_EQ(point.x, i * 1);
        EXPECT_DOUBLE_EQ(point.y, i * 2);
        EXPECT_DOUBLE_EQ(point.z, i * 3);
    }

    //EXPECT_DOUBLE_EQ(glcm.at<float>(1, 1), 1.0);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

