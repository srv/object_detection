#ifndef TRANSFORMATION_CALCULATOR_H
#define TRANSFORMATION_CALCULATOR_H

#include <vector>

namespace cv {
    template <class T> class Point3_;
    typedef Point3_<float> Point3f;
    class Mat;
}

namespace object_detection
{

/**
* \class TransformationCalculator
* \author Stephan Wirth
* \brief calculates the transformation between two 3D point sets
*/
class TransformationCalculator
{

  public:
    /**
    * Calculates the transformation matrix to transform points1 to
    * points2. Uses PCL's estimateRigidTransformationSVD
    */
    static void calculateRigidTransformation(
        const std::vector<cv::Point3f>& points1,
        const std::vector<cv::Point3f>& points2, cv::Mat& transform);
};

}

#endif

