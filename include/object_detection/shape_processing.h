#ifndef SHAPE_PROCESSING_H
#define SHAPE_PROCESSING_H

#include <opencv2/core/core.hpp>

namespace object_detection
{
  namespace shape_processing
  {

    typedef std::vector<cv::Point> Shape;

    /**
    * Computes the area of a shape
    */
    double area(const Shape& shape);

    /**
    * gets the minimal bounding axis aligned rectangle
    */
    cv::Rect boundingRect(const Shape& shape);

    /**
    * gets a mask image of size boundingRect() that contains 0 for
    * background and 255 for everything in the shape
    */
    cv::Mat minimalMask(const Shape& shape);

    /**
    * shifts all points of the given shape by given parameters and returns
    * the newly created shape
    */
    Shape shift(const Shape& shape, float dx, float dy);
      
    /**
    * Comparison method for shape sorting.
    * After sorting with this comparator the biggest shape comes first.
    */
    bool compareShapeArea(const Shape& shape1, const Shape& shape2);

    /**
    * Extracts shapes from given binary image
    */
    std::vector<Shape> extractShapes(const cv::Mat& image);

    /**
    * Filters input shapes and returns a set with all shapes that are at least
    * as big as half the size of the biggest shape.
    */
    std::vector<Shape> getBiggestShapes(const std::vector<Shape>& shapes);
  }

} // end of namespace


#endif

