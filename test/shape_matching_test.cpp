#include <gtest/gtest.h>
#include <cv.h>

#include "object_detection/shape_matching.h"

using namespace object_detection;

TEST(ShapeMatching, computeCentroid1)
{
    std::vector<cv::Point> polygon;
    polygon.push_back(cv::Point( 10,  10));
    polygon.push_back(cv::Point(-10,  10));
    polygon.push_back(cv::Point(-10, -10));
    polygon.push_back(cv::Point( 10, -10));

    cv::Moments moments = cv::moments(cv::Mat(polygon));

    cv::Point centroid = ShapeMatching::computeCentroid(moments);

    EXPECT_EQ(centroid.x, 0);
    EXPECT_EQ(centroid.y, 0);
}

TEST(ShapeMatching, computeCentroid2)
{
    std::vector<cv::Point> polygon;
    polygon.push_back(cv::Point( 20, 10));
    polygon.push_back(cv::Point(120, 10));
    polygon.push_back(cv::Point(120, 20));
    polygon.push_back(cv::Point( 20, 20));

    cv::Moments moments = cv::moments(cv::Mat(polygon));

    cv::Point centroid = ShapeMatching::computeCentroid(moments);

    EXPECT_EQ(centroid.x, 70);
    EXPECT_EQ(centroid.y, 15);
}

TEST(ShapeMatching, findRotation1)
{
    std::vector<cv::Point> shape1;
    shape1.push_back(cv::Point(  0, -30));
    shape1.push_back(cv::Point(-10,  10));
    shape1.push_back(cv::Point( 10,  10));

    std::vector<cv::Point> shape2;
    shape2.push_back(cv::Point( 30,   0));
    shape2.push_back(cv::Point(-10, -10));
    shape2.push_back(cv::Point(-10,  10));

    double score;

    EXPECT_DOUBLE_EQ(ShapeMatching::findRotation(shape1, shape2, &score),  M_PI_2);
    EXPECT_DOUBLE_EQ(score, 1.0);
    EXPECT_DOUBLE_EQ(ShapeMatching::findRotation(shape2, shape1, &score), -M_PI_2);
    EXPECT_DOUBLE_EQ(score, 1.0);

    EXPECT_DOUBLE_EQ(ShapeMatching::findRotation(shape1, shape1, &score), 0);
    EXPECT_DOUBLE_EQ(score, 1.0);
    EXPECT_DOUBLE_EQ(ShapeMatching::findRotation(shape2, shape2, &score), 0);
    EXPECT_DOUBLE_EQ(score, 1.0);
}

TEST(ShapeMatching, rotatePoints)
{   
    std::vector<cv::Point> points;
    points.push_back(cv::Point(100, 200));

    std::vector<cv::Point> rotated_points;
    rotated_points = ShapeMatching::rotatePoints(points, 0.0);

    ASSERT_EQ(rotated_points.size(), points.size());

    EXPECT_EQ(rotated_points[0].x, 100);
    EXPECT_EQ(rotated_points[0].y, 200);

    rotated_points = ShapeMatching::rotatePoints(points, M_PI_2);
    EXPECT_EQ(rotated_points[0].x, -200);
    EXPECT_EQ(rotated_points[0].y, 100);


    // create some random points
    // for forth and back rotation test
    cv::Mat mat(100, 1, CV_32SC2);
    cv::randu(mat, cv::Scalar(0, 0), cv::Scalar(2000,3000));
    points = cv::Mat_<cv::Point>(mat);

    rotated_points = ShapeMatching::rotatePoints(points, M_PI_2 / 2);

    std::vector<cv::Point> rotated_points2;
    rotated_points2 = ShapeMatching::rotatePoints(rotated_points, -M_PI_2 / 2);

    ASSERT_EQ(points.size(), rotated_points2.size());

    for (size_t i = 0; i < points.size(); ++i)
    {
        EXPECT_LE(abs(points[i].x - rotated_points2[i].x), 2);
        EXPECT_LE(abs(points[i].y - rotated_points2[i].y), 2);
    }
}


TEST(ShapeMatching, computeIntersectionArea)
{
    std::vector<cv::Point> shape1;
    shape1.push_back(cv::Point(  0,   0));
    shape1.push_back(cv::Point( 30,   0));
    shape1.push_back(cv::Point( 30,  30));
    shape1.push_back(cv::Point(  0,  30));

    std::vector<cv::Point> shape2;
    shape2.push_back(cv::Point( 10,  10));
    shape2.push_back(cv::Point( 50,  10));
    shape2.push_back(cv::Point( 50,  40));
    shape2.push_back(cv::Point( 10,  40));

    double area = ShapeMatching::computeIntersectionArea(shape1, shape2);

    EXPECT_DOUBLE_EQ(area, 400.0);

    // intersect circle with rect
    shape1.clear();
    double radius = 1000;
    for (int i = 0; i < 360; ++i)
    {
        int x = static_cast<int>(round(sin(i / 180.0 * M_PI) * radius));
        int y = static_cast<int>(round(cos(i / 180.0 * M_PI) * radius));
        shape1.push_back(cv::Point(x, y));
    }
    
    shape2.clear();
    shape2.push_back(cv::Point(0, 0));
    shape2.push_back(cv::Point(0, radius));
    shape2.push_back(cv::Point(radius, radius));
    shape2.push_back(cv::Point(radius, 0));

    area = ShapeMatching::computeIntersectionArea(shape1, shape2);

    EXPECT_NEAR(area, (M_PI * radius * radius)/4.0, 100);

}

TEST(ShapeMatching, matchShapes)
{
    std::vector<cv::Point> shape1;
    shape1.push_back(cv::Point( 100,  200));
    shape1.push_back(cv::Point(-100,  200));
    shape1.push_back(cv::Point(-100, -200));
    shape1.push_back(cv::Point(   0, -300));
    shape1.push_back(cv::Point( 100, -200));

    double angle = M_PI / 4.0;
    double scale = 1.5;
    int shift_x =  50;
    int shift_y = 150;
    std::vector<cv::Point> shape2 = ShapeMatching::rotatePoints(shape1, M_PI / 4.0);
    for (size_t i = 0; i < shape2.size(); ++i)
    {
        shape2[i].x = shape2[i].x * scale + shift_x;
        shape2[i].y = shape2[i].y * scale + shift_y;
    }

    double score;
    ShapeMatching::MatchingParameters matching_parameters =
        ShapeMatching::matchShapes(shape2, shape1, &score);

    EXPECT_NEAR(score, 1.0, 0.01);
    EXPECT_NEAR(-shift_x, matching_parameters.shift_x, 2);
    EXPECT_NEAR(-shift_y, matching_parameters.shift_y, 2);
    EXPECT_NEAR(-angle, matching_parameters.rotation, M_PI / 180.0);
    EXPECT_NEAR(1.0/scale, matching_parameters.scale, 0.1);
}

TEST(ShapeMatching, matchShapes2)
{
    std::vector<cv::Point> ref_shape;
    ref_shape.push_back(cv::Point(500, 100));
    ref_shape.push_back(cv::Point(300, 200));
    ref_shape.push_back(cv::Point(300,-200));
    ref_shape.push_back(cv::Point(500,-100));

    std::vector<cv::Point> detected_shape;
    detected_shape.push_back(cv::Point(300,  100));
    detected_shape.push_back(cv::Point(400,  300));
    detected_shape.push_back(cv::Point(  0,  300));
    detected_shape.push_back(cv::Point(100,  100));

    double score;
    ShapeMatching::MatchingParameters matching_parameters =
        ShapeMatching::matchShapes(detected_shape, ref_shape, &score);

    std::cout << "Matching parameters: " << std::endl;
    std::cout << "  rotation (deg.):" << matching_parameters.rotation / M_PI * 180.0 << std::endl;
    std::cout << "  shift_x        :" << matching_parameters.shift_x << std::endl;
    std::cout << "  shift_y        :" << matching_parameters.shift_y << std::endl;
    std::cout << "  scale          :" << matching_parameters.scale << std::endl;
    std::cout << "  score          :" << score << std::endl;
    
    EXPECT_NEAR(score, 1.0, 0.02);
    EXPECT_NEAR(matching_parameters.shift_x, -200, 2);
    EXPECT_NEAR(matching_parameters.shift_y, -600, 2);
    EXPECT_NEAR(M_PI / 2, matching_parameters.rotation, M_PI / 180.0);
    EXPECT_NEAR(1.0, matching_parameters.scale, 0.1);
}

TEST(ShapeMatching, matchShapes3)
{
    std::vector<cv::Point> ref_shape;
    ref_shape.push_back(cv::Point(100, 100));
    ref_shape.push_back(cv::Point(100,-100));
    ref_shape.push_back(cv::Point(200,   0));

    std::vector<cv::Point> detected_shape;
    detected_shape.push_back(cv::Point(  0,  400));
    detected_shape.push_back(cv::Point(200,  200));
    detected_shape.push_back(cv::Point(400,  400));

    double score;
    ShapeMatching::MatchingParameters matching_parameters =
        ShapeMatching::matchShapes(detected_shape, ref_shape, &score);

    std::cout << "Matching parameters: " << std::endl;
    std::cout << "  rotation (deg.):" << matching_parameters.rotation / M_PI * 180.0 << std::endl;
    std::cout << "  shift_x        :" << matching_parameters.shift_x << std::endl;
    std::cout << "  shift_y        :" << matching_parameters.shift_y << std::endl;
    std::cout << "  scale          :" << matching_parameters.scale << std::endl;
    std::cout << "  score          :" << score << std::endl;
    
    EXPECT_NEAR(score, 1.0, 0.1);
    EXPECT_NEAR(matching_parameters.shift_x, -200, 2);
    EXPECT_NEAR(matching_parameters.shift_y, -600, 2);
    EXPECT_NEAR(M_PI_2, matching_parameters.rotation, M_PI / 180.0);
    EXPECT_NEAR(0.5, matching_parameters.scale, 0.1);

}


TEST(ShapeMatching, computeMeanDistance)
{
    std::vector<cv::Point> shape1;
    shape1.push_back(cv::Point( 10,  20));
    shape1.push_back(cv::Point(-10,  20));
    shape1.push_back(cv::Point(-10, -20));
    shape1.push_back(cv::Point( 10, -20));

    double distance = ShapeMatching::computeMeanDistance(shape1, cv::Point(0, 0));
    EXPECT_DOUBLE_EQ(distance, sqrt(500));
}




// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

