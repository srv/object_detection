
#include <iostream>
#include <cv.h>

#include "object_detection/clipper.hpp"
#include "object_detection/shape_matching.h"


using namespace object_detection;

ShapeMatching::MatchingParameters ShapeMatching::matchShapes(
        const std::vector<cv::Point>& floating_shape,
        const std::vector<cv::Point>& reference_shape,
        double* score)
{
    cv::Moments reference_shape_moments = cv::moments(cv::Mat(reference_shape));
    cv::Moments floating_shape_moments = cv::moments(cv::Mat(floating_shape));

    cv::Point reference_shape_centroid = computeCentroid(reference_shape_moments);
    cv::Point floating_shape_centroid = computeCentroid(floating_shape_moments);
 
    double reference_shape_mean_distance = computeMeanDistance(reference_shape,
            reference_shape_centroid);
    double floating_shape_mean_distance = computeMeanDistance(floating_shape,
            floating_shape_centroid);

    /*
    std::cout << "reference shape: area=" << reference_shape_area 
              << " centroid=" << reference_shape_centroid 
              << " mean distance=" << reference_shape_mean_distance << std::endl;
    std::cout << "floating shape: area=" << floating_shape_area 
              << " centroid=" << floating_shape_centroid 
              << " mean distance=" << floating_shape_mean_distance << std::endl;
              */

    double scale = reference_shape_mean_distance / floating_shape_mean_distance;

    // normalize shapes
    std::vector<cv::Point> normalized_reference_shape(reference_shape.size());
    for(size_t i = 0; i < normalized_reference_shape.size(); ++i)
    {
        normalized_reference_shape[i] = reference_shape[i] -
            reference_shape_centroid;
    }
    std::vector<cv::Point> normalized_floating_shape(floating_shape.size());
    for(size_t i = 0; i < normalized_floating_shape.size(); ++i)
    {
        normalized_floating_shape[i] = (floating_shape[i] -
            floating_shape_centroid) * scale;
    }

    MatchingParameters parameters;
    parameters.scale = scale;
    parameters.shift_x = reference_shape_centroid.x - floating_shape_centroid.x;
    parameters.shift_y = reference_shape_centroid.y - floating_shape_centroid.y;
    parameters.rotation = findRotation(normalized_floating_shape,
            normalized_reference_shape, score);
    return parameters;
}

cv::Point ShapeMatching::computeCentroid(const cv::Moments& moments)
{
    cv::Point centroid;
    centroid.x = moments.m10 / moments.m00;
    centroid.y = moments.m01 / moments.m00;
    return centroid;
}


double ShapeMatching::findRotation(const std::vector<cv::Point>& floating_shape,
        const std::vector<cv::Point>& reference_shape, double* score)
{
    double floating_shape_area = cv::contourArea(cv::Mat(floating_shape));
    double reference_shape_area = cv::contourArea(cv::Mat(reference_shape));

    int best_angle = 0;
    double max_score = 0.0;
    for (int angle = -179; angle <= 180; ++angle)
    {
        std::vector<cv::Point> rotated_shape = rotatePoints(floating_shape, angle / 180.0 * M_PI);
        double intersection_area = 
            computeIntersectionArea(rotated_shape, reference_shape);
        double union_area = floating_shape_area + reference_shape_area -
            intersection_area;

        if (union_area > 0.00001)
        {
            double current_score = intersection_area / union_area;
            if (current_score > max_score)
            {
                max_score = current_score;
                best_angle = angle;
            }
        }
    }
    if (score != NULL) *score = max_score;
    return best_angle / 180.0 * M_PI;
}

std::vector<cv::Point> ShapeMatching::rotatePoints(
        const std::vector<cv::Point>& points, double angle)
{
    double sinAngle = sin(-angle);
    double cosAngle = cos(-angle);
    std::vector<cv::Point> rotated_points(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
        rotated_points[i].x = round(cosAngle * points[i].x + sinAngle * points[i].y);
        rotated_points[i].y = round(-sinAngle * points[i].x + cosAngle * points[i].y);
    }
    return rotated_points;
}

double ShapeMatching::computeIntersectionArea(
        const std::vector<cv::Point>& shape1,
        const std::vector<cv::Point>& shape2)
{

    // convert to clipper data structures
    clipper::Polygon poly1(shape1.size());
    for (size_t i = 0; i < shape1.size(); ++i)
    {
        poly1[i] = clipper::IntPoint(shape1[i].x, shape1[i].y);
    }
    clipper::Polygon poly2(shape2.size());
    for (size_t i = 0; i < shape2.size(); ++i)
    {
        poly2[i] = clipper::IntPoint(shape2[i].x, shape2[i].y);
    }

    clipper::Clipper clipper;
    clipper.AddPolygon(poly1, clipper::ptSubject);
    clipper.AddPolygon(poly2, clipper::ptClip);

    clipper::Polygons intersection_polygons;
    if (clipper.Execute(clipper::ctIntersection, intersection_polygons))
    {
        // convert back to opencv to compute area
        double area = 0.0;
        for (size_t i = 0; i < intersection_polygons.size(); ++i)
        {
            std::vector<cv::Point> cv_poly(intersection_polygons[i].size());
            for (size_t j = 0; j < cv_poly.size(); ++j)
            {
                cv_poly[j] = cv::Point(intersection_polygons[i][j].X,
                        intersection_polygons[i][j].Y);
            }
            area += cv::contourArea(cv::Mat(cv_poly));
        }
        return area;
    }
    return 0.0;
}

double ShapeMatching::computeMeanDistance(
        const std::vector<cv::Point>& points, const cv::Point& reference_point)
{
    assert(points.size() != 0);

    double distance_sum = 0.0;
    for (size_t i = 0; i < points.size(); ++i)
    {
        double x_diff = points[i].x - reference_point.x;
        double y_diff = points[i].y - reference_point.y;
        double distance = sqrt(x_diff*x_diff + y_diff*y_diff);
        distance_sum += distance;
    }
    return distance_sum / points.size();
}


