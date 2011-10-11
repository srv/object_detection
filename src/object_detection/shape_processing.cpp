#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "object_detection/shape_processing.h"

double object_detection::shape_processing::area(const Shape& shape)
{
  return cv::contourArea(cv::Mat(shape));
}

cv::Rect object_detection::shape_processing::boundingRect(const Shape& shape)
{
  return cv::minAreaRect(cv::Mat(shape)).boundingRect();
}

cv::Mat object_detection::shape_processing::minimalMask(const Shape& shape)
{
  cv::Rect bounding_rect = boundingRect(shape);
  cv::Mat mask = cv::Mat::zeros(bounding_rect.height, bounding_rect.width, CV_8UC1);
  std::vector<Shape> shapes;
  shapes.push_back(shape);
  cv::drawContours(mask, shapes, 0, cv::Scalar::all(255), CV_FILLED, 8, std::vector<cv::Vec4i>(), INT_MAX, -bounding_rect.tl());
  return mask;
}

object_detection::shape_processing::Shape object_detection::shape_processing::shift(const Shape& shape, float dx, float dy)
{
  Shape shifted_shape = shape;
  for (size_t i = 0; i < shifted_shape.size(); ++i)
  {
    shifted_shape[i].x += dx;
    shifted_shape[i].y += dy;
  }
  return shifted_shape;
}

bool object_detection::shape_processing::compareShapeArea(const Shape& shape1, const Shape& shape2)
{
    return area(shape1) > area(shape2);
}

std::vector<object_detection::shape_processing::Shape> object_detection::shape_processing::extractShapes(const cv::Mat& image)
{
  cv::Mat image_copy;
  image.convertTo(image_copy, CV_8UC1);
  std::vector<Shape> contours;
  //cv::findContours(image_copy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  cv::findContours(image_copy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);
  std::vector<Shape> simplified_contours;
  for (size_t i = 0; i < contours.size(); ++i)
  {
    Shape simplified;
    cv::approxPolyDP(contours[i], simplified, 5, true); // allow 5 pixel displacement
    if (simplified.size() > 2)
    {
      simplified_contours.push_back(simplified);
    }
  }
  return simplified_contours;
}

std::vector<object_detection::shape_processing::Shape> object_detection::shape_processing::getBiggestShapes(const std::vector<Shape>& shapes)
{
  std::vector<Shape> sorted_shapes = shapes;
  std::sort(sorted_shapes.begin(), sorted_shapes.end(), compareShapeArea);
  std::vector<std::vector<cv::Point> > biggest_shapes;
  if (sorted_shapes.size() > 0)
  {
    double biggest_area = cv::contourArea(cv::Mat(sorted_shapes[0]));
    std::vector<Shape>::const_iterator iter = 
        sorted_shapes.begin();
    bool stop = false;
    while (!stop && iter != sorted_shapes.end())
    {
      if (cv::contourArea(cv::Mat(*iter)) > 0.5 * biggest_area)
      {
        biggest_shapes.push_back(*iter);
      }
      else
      {
        stop = true;
      }
      ++iter;
    }
  }
  return biggest_shapes;
}

