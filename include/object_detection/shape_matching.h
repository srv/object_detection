#ifndef SHAPE_MATCHING_H 
#define SHAPE_MATCHING_H

namespace cv {
    class Mat;
    template<typename T> class Point_;
    typedef Point_<int> Point;
    class Moments;
}

namespace object_detection {

/**
 * \class ShapeMatching
 * \author Stephan Wirth
 * \brief Simple matching of shapes.
 */
class ShapeMatching
{
public:

    struct MatchingParameters
    {
        double scale;
        double rotation;
        int shift_x;
        int shift_y;
    };

    /**
    * \brief Finds the optimal transformation to match two shapes.
    * The transformation is estimated following these steps:
    * -# The input shapes are normalized using normalizeShape.
    * -# The optimal rotation between floating_shape and reference_shape
    *    is computed using findRotation
    *
    * \param floating_shape the shape to match on the reference_shape
    * \param reference_shape the reference shape outline
    * \param score if != NULL, the score that findRotation computes
    *        will be stored here
    * \return the parameters that led to an optimal match
    */
    static MatchingParameters matchShapes(
            const std::vector<cv::Point>& floating_shape,
            const std::vector<cv::Point>& reference_shape, double* score);


    /**
    * Calculates the centroid of a shape using its moments and returns it.
    * \param moments input moments
    * \return the centroid
    */
    static cv::Point computeCentroid(const cv::Moments& moments);

    /**
    * Finds the rotation that results in maximal overlap of two shapes.
    * \param floating_shape the floating shape that has to be rotated
    * \param reference_shape the reference shape
    * \param score if != NULL. the best matching score will be stored here.
    *        1 means perfect match, 0 means no match.
    * \return the angle that has to be used to rotate floating_shape
    *         to result in a maximal overlap with reference_shape in radiants.
    */
    static double findRotation(const std::vector<cv::Point>& floating_shape,
            const std::vector<cv::Point>& reference_shape, double* score = NULL);

    /**
    * Rotates a set of image points clock-wise (as rotation goes from x to y)
    * by a given angle and shifts the result afterwards.
    * \param points the input points
    * \param angle the angle to use for rotation in radiants
    * \param shift_x shift in x direction
    * \param shift_y shift in y direction
    * \return rotated point set
    */
    static std::vector<cv::Point> rotatePoints(
            const std::vector<cv::Point>& points, 
            double angle, int shift_x = 0, int shift_y = 0);

    /**
    * Computes the area that two shapes have in common.
    * \param shape1 the first shape
    * \param shape2 the second shape
    * \return area of the region that forms the intersection of shape1 and shape2
    */
    static double computeIntersectionArea(const std::vector<cv::Point>& shape1,
        const std::vector<cv::Point>& shape2);

    /**
    * Computes the mean distance of a set of points with respect to a reference
    * point.
    * \param points the set of points, must have size() != 0
    * \param reference_point the reference point
    */
    static double computeMeanDistance(const std::vector<cv::Point>& points, 
            const cv::Point& reference_point);


};

}


#endif /* SHAPE_MATCHING_H */

