#include <gtest/gtest.h>
#include <cv.h>

#include "glcm.h"

using namespace object_detection;

TEST(GLCM, precondition_assertions)
{
    cv::Mat image(600, 800, CV_8UC1);
    cv::randu(image, cv::Scalar(0), cv::Scalar(256));

    EXPECT_DEATH(computeGLCM(image, 0, 0), ".*Assert.*");
    EXPECT_DEATH(computeGLCM(image, -1, 0),".*Assert.*");
    EXPECT_DEATH(computeGLCM(image, 0, -1),".*Assert.*");
    EXPECT_DEATH(computeGLCM(image, 1, 0, -1),".*Assert.*");
    EXPECT_DEATH(computeGLCM(image, 1, 0, 1),".*Assert.*");
    EXPECT_DEATH(computeGLCM(image, 1, 0, 3),".*Assert.*");
    EXPECT_DEATH(computeGLCM(image, 1, 0, 15),".*Assert.*");
    EXPECT_DEATH(computeGLCM(image, 800, 0),".*Assert.*");
    EXPECT_DEATH(computeGLCM(image, 0, 600),".*Assert.*");

    cv::Mat imagec3(600, 600, CV_8UC3);
    EXPECT_DEATH(computeGLCM(imagec3, 1, 1),".*Assert.*");
}

TEST(GLCM, sizes)
{
    cv::Mat image(600, 800, CV_8UC1);
    cv::randu(image, cv::Scalar(0), cv::Scalar(256));
    for (int i = 2; i <= 256; i*=2)
    {
        cv::Mat glcm = computeGLCM(image, 1, 0, i);
        EXPECT_EQ(glcm.rows, i);
        EXPECT_EQ(glcm.cols, i);
    }
    cv::Mat glcm = computeGLCM(image, 1, 0);
    EXPECT_EQ(glcm.rows, 256);
    EXPECT_EQ(glcm.cols, glcm.rows);
    EXPECT_EQ(glcm.channels(), 1);
    EXPECT_EQ(glcm.depth(), CV_32F);
}

TEST(GLCM, symmetry)
{
    cv::Mat image(600, 800, CV_8UC1);
    cv::randu(image, cv::Scalar(0), cv::Scalar(256));
    cv::Mat glcm = computeGLCM(image, 1, 0);
    for (int r = 0; r < glcm.rows; ++r)
    {
        for (int c = 0; c < glcm.cols; ++c)
        {
            EXPECT_DOUBLE_EQ(glcm.at<float>(r, c),
                             glcm.at<float>(c, r));
        }
    }
}

TEST(GLCM, norm)
{
    cv::Mat image(600, 800, CV_8UC1);
    cv::randu(image, cv::Scalar(0), cv::Scalar(256));
    cv::Mat glcm = computeGLCM(image, 1, 0);
    double sum = 0.0;
    for (int r = 0; r < glcm.rows; ++r)
    {
        for (int c = 0; c < glcm.cols; ++c)
        {
            sum += glcm.at<float>(c, r);
        }
    }
    EXPECT_NEAR(sum, 1.0, 0.00001);
}

TEST(GLCM, computeGLCM)
{
    unsigned char data[] =
        { 0, 0, 0, 
          0, 0, 0,
          0, 0, 0 };
    cv::Mat image = cv::Mat(3, 3, CV_8UC1, data).clone();
    cv::Mat glcm = computeGLCM(image, 1, 0);
    for (int r = 0; r < glcm.rows; ++r)
    {
        for (int c = 0; c < glcm.cols; ++c)
        {
            if (r == 0 && c == 0)
            {
                EXPECT_DOUBLE_EQ(glcm.at<float>(r, c), 1.0);
            }
            else
            {
                EXPECT_DOUBLE_EQ(glcm.at<float>(r, c), 0.0);
            }
        }
    }
    cv::Scalar features = computeGLCMFeatures(glcm);
    EXPECT_DOUBLE_EQ(features[0], 0.0);
    EXPECT_DOUBLE_EQ(features[1], 1.0);
}

TEST(GLCM, computeGLCM2)
{
    unsigned char data[] =
        { 1, 1, 1, 
          1, 1, 1,
          1, 1, 1 };
    cv::Mat image = cv::Mat(3, 3, CV_8UC1, data).clone();
    cv::Mat glcm = computeGLCM(image, 1, 0);
    EXPECT_DOUBLE_EQ(glcm.at<float>(1, 1), 1.0);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

