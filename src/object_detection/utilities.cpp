#include "utilities.h"

std::vector<cv::Point> object_detection::computeRectanglePoints(const cv::RotatedRect& rotated_rect)
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
        cv::getRotationMatrix2D(cv::Point2f(0.0, 0.0), -rotated_rect.angle, scale);

    cv::Mat rotated_points_matrix;
    cv::transform(points_matrix, rotated_points_matrix, rotation_matrix);
    cv::add(rotated_points_matrix, 
            cv::Scalar(rotated_rect.center.x, rotated_rect.center.y), 
            rotated_points_matrix);

    std::vector<cv::Point> rotated_points = cv::Mat_<cv::Point>(rotated_points_matrix);
    return rotated_points;
}

void object_detection::paintRotatedRectangle(cv::Mat& img, const cv::RotatedRect& rect,
       const cv::Scalar& color, int thickness, int lineType, int shift)
{
    std::vector<cv::Point> rectangle_points = computeRectanglePoints(rect);
    bool is_closed = true;
    const cv::Point* points[1];
    points[0] = rectangle_points.data();
    int size = rectangle_points.size();
    cv::polylines(img, points, &size, 1, is_closed, color, thickness, lineType, shift);
}

void object_detection::paintFilledRotatedRectangle(cv::Mat& img, const cv::RotatedRect& rect,
       const cv::Scalar& color, int lineType, int shift)
{
    std::vector<cv::Point> rectangle_points = computeRectanglePoints(rect);
    const cv::Point* points = rectangle_points.data();
    int size = rectangle_points.size();
    cv::fillConvexPoly(img, points, size, color, lineType, shift);
}

void object_detection::paintPolygon(cv::Mat& img, const std::vector<cv::Point>& points,
       const cv::Scalar& color, int thickness, int lineType, int shift)
{
    bool is_closed = true;
    const cv::Point* point_data = points.data();
    int size = points.size();
    cv::polylines(img, &point_data, &size, 1, is_closed, color, thickness, lineType, shift);
}

void object_detection::paintFilledPolygon(cv::Mat& img, const std::vector<cv::Point>& points,
       const cv::Scalar& color, int lineType, int shift)
{
    const cv::Point* point_data = points.data();
    int size = points.size();
    cv::fillPoly(img, &point_data, &size, 1, color, lineType, shift);
}

