#include <stdexcept>

#include <ros/package.h>

#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "object_detection/shape_processing.h"

using namespace object_detection;
using shape_processing::Shape;

cv::Mat loadImage(const std::string& name)
{
  std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
  cv::Mat image = cv::imread(path + "/data/" + name, 0);
  if (image.empty())
  {
    throw std::runtime_error("did not find image " + name);
  }
  return image;
}

TEST(ShapeProcessing, rawTime)
{
  cv::Mat image = loadImage("complex_blobs.jpg");
  std::vector<Shape> shapes;
  double time = (double)cv::getTickCount();
  cv::findContours(image, shapes, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);
  time = ((double)cv::getTickCount() - time)/cv::getTickFrequency() * 1000;
  std::cout << "findContours took " << time << "ms." << std::endl;
}

TEST(ShapeProcessing, extractShapes)
{
  cv::Mat image = loadImage("sharp_mask.jpg");
  std::vector<Shape> shapes = shape_processing::extractShapes(image);
  EXPECT_EQ(shapes.size(), 1);
}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

