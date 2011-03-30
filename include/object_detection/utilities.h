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
std::vector<cv::Point> computeRectanglePoints(const cv::RotatedRect& rotated_rect)
{
    int x = rotated_rect.size.width / 2;
    int y = rotated_rect.size.height / 2;
    std::vector<cv::Point2f> points(4);
    points[0] = cv::Point2f( x,  y);
    points[1] = cv::Point2f(-x,  y);
    points[2] = cv::Point2f(-x, -y);
    points[3] = cv::Point2f( x, -y);

    cv::Mat points_matrix(points); // points_mat is now a 4x1 CV_32FC2 image

    const float scale = 1.0;
    cv::Mat rotation_matrix = 
        cv::getRotationMatrix2D(cv::Point2f(0.0, 0.0), rotated_rect.angle, scale);

    cv::Mat rotated_points_matrix;
    cv::transform(points_matrix, rotated_points_matrix, rotation_matrix);
    cv::add(rotated_points_matrix, 
            cv::Scalar(rotated_rect.center.x, rotated_rect.center.y), 
            rotated_points_matrix);

    std::vector<cv::Point> rotated_points = cv::Mat_<cv::Point>(rotated_points_matrix);
    return rotated_points;
}

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
       const cv::Scalar& color, int thickness=1, int lineType=8, int shift=0)
{
    std::vector<cv::Point> rectangle_points = computeRectanglePoints(rect);
    bool is_closed = true;
    const cv::Point* points[1];
    points[0] = rectangle_points.data();
    int size = rectangle_points.size();
    cv::polylines(img, points, &size, 1, is_closed, color, thickness, lineType, shift);
}

/**
 * Paints a filled rotated rectangle on given image using cv::fillConvexPoly
 * \param image the image to draw on
 * \param rect the rectangle to draw
 * \param color the color to use
 * \param lineType type of line (passed to cv::fillConvexPoly)
 * \param shift shifting of the drawing (passed to cv::fillConvexPoly)
 */
void paintFilledRotatedRectangle(cv::Mat& img, const cv::RotatedRect& rect,
       const cv::Scalar& color, int lineType=8, int shift=0)
{
    std::vector<cv::Point> rectangle_points = computeRectanglePoints(rect);
    const cv::Point* points = rectangle_points.data();
    int size = rectangle_points.size();
    cv::fillConvexPoly(img, points, size, color, lineType, shift);
}



} // namespace object_detection

#endif

