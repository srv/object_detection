#include <gtest/gtest.h>

#include "detector.h"
#include "model.h"

#include <cstdlib>

using namespace object_detection;


void checkMatrixEqual(const cv::Mat& mat1, const cv::Mat& mat2, double epsilon)
{
    ASSERT_EQ(mat1.rows, mat2.rows);
    ASSERT_EQ(mat1.cols, mat2.cols);
    ASSERT_EQ(mat1.type(), mat2.type());
    ASSERT_EQ(mat1.type(), CV_64F); // we test only float

    for (int r = 0; r < mat1.rows; ++r)
    {
        for (int c = 0; c < mat1.cols; ++c)
        {
            EXPECT_NEAR(mat1.at<double>(r, c), mat2.at<double>(r, c), epsilon);
        }
    }
}


TEST(Detector, empty)
{
    Model object_model;
    Model scene_model;

    cv::Mat transformation;
    bool success = Detector::estimatePose(object_model, scene_model, transformation);
    EXPECT_EQ(success, false);
}

TEST(Detector, pose_estimation)
{
    srand(time(NULL));
    Model object_model;
    Model scene_model;
    for (int i = 0; i < 10; ++i)
    {
        cv::Point3f point(rand(), rand(), rand());
        Feature feature;
        feature.descriptor.resize(64);
        for (size_t j = 0; j < feature.descriptor.size(); ++j)
        {
            feature.descriptor[j] = rand();
        }
        object_model.addFeature(point, feature);

        scene_model.addFeature(point, feature);
    }

    cv::Mat transformation;
    bool success = Detector::estimatePose(object_model, scene_model, transformation);
    EXPECT_EQ(success, true);

    cv::Mat identity = cv::Mat::eye(3, 4, CV_64F);

    double epsilon = 1e-4;
    checkMatrixEqual(transformation, identity, epsilon);
}

TEST(Detector, pose_estimation_translation)
{
    srand(time(NULL));
    Model object_model;
    Model scene_model;

    cv::Mat transform_mat = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat last_col = transform_mat.col(3);
    randu(last_col, cv::Scalar(-0.5), cv::Scalar(0.5));

    int num_points = 10;
    cv::Mat points(num_points, 1, CV_64FC3);
    randu(points, cv::Scalar(-0.5, -0.5, 0.5), cv::Scalar(0.5, 0.5, 2.5));

    std::cout << points << std::endl;
    cv::transform(points, points, transform_mat);
    std::cout << "transformed:" << std::endl;
    std::cout << points << std::endl;

    /*
    for (int i = 0; i < num_points; ++i)
    {
        cv::Point3f point(rand(), rand(), rand());
        Feature feature;
        feature.descriptor.resize(64);
        for (size_t j = 0; j < feature.descriptor.size(); ++j)
        {
            feature.descriptor[j] = rand();
        }
        object_model.addFeature(point, feature);

        scene_model.addFeature(transform_mat * point, feature);
    }

    cv::Mat transformation;
    bool success = Detector::estimatePose(object_model, scene_model, transformation);
    EXPECT_EQ(success, true);


    double epsilon = 1e-4;
    checkMatrixEqual(transformation, identity, epsilon);
    */
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

