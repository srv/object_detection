#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>

#include <vector>
#include <cv.h>

namespace object_detection {

/**
 * computes the points of a rotated rectangle
 * \param rotated_rect the rectangle to compute the corner points for
 * \return a vector containing the rectangle points
 */
std::vector<cv::Point> computeRectanglePoints(const cv::RotatedRect& rotated_rect);

/**
 * Paints a rotated rectangle on given image using cv::polylines
 * \param image the image to draw on
 * \param rect the rectangle to draw
 * \param color the color to use
 * \param thickness the thickness of the line
 * \param lineType type of line (passed to cv::polylines)
 * \param shift shifting of the drawing (passed to cv::polylines)
 */
void paintRotatedRectangle(cv::Mat& img, const cv::RotatedRect& rect,
       const cv::Scalar& color, int thickness=1, int lineType=8, int shift=0);

/**
 * Paints a filled rotated rectangle on given image using cv::fillConvexPoly
 * \param image the image to draw on
 * \param rect the rectangle to draw
 * \param color the color to use
 * \param lineType type of line (passed to cv::fillConvexPoly)
 * \param shift shifting of the drawing (passed to cv::fillConvexPoly)
 */
void paintFilledRotatedRectangle(cv::Mat& img, const cv::RotatedRect& rect,
       const cv::Scalar& color, int lineType=8, int shift=0);

/**
 * Paints a polygon on given image using cv::polylines
 * \param image the image to draw on
 * \param points the polygon outline points
 * \param color the color to use
 * \param thickness the thickness of the line
 * \param lineType type of line (passed to cv::polylines)
 * \param shift shifting of the drawing (passed to cv::polylines)
 */
void paintPolygon(cv::Mat& img, const std::vector<cv::Point>& points,
       const cv::Scalar& color, int thickness=1, int lineType=8, int shift=0);

/**
 * Paints a filled polygon on given image using cv::fillPoly
 * \param image the image to draw on
 * \param points the polygon outline points
 * \param color the color to use
 * \param lineType type of line (passed to cv::fillConvexPoly)
 * \param shift shifting of the drawing (passed to cv::fillConvexPoly)
 */
void paintFilledPolygon(cv::Mat& img, const std::vector<cv::Point>& points,
       const cv::Scalar& color, int lineType=8, int shift=0);

} // namespace object_detection

#endif

