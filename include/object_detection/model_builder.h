#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include <vector>

#include <opencv2/core/core.hpp>
namespace object_detection {

class Feature;
class Model;

/**
 * \class ModelBuilder
 * \author Stephan Wirth
 * \brief Builds a model
 */
class ModelBuilder
{

  public:

    /**
    * Extend the given model by matching given features and adding
    * those features/points that are new
    */
    static void extend(Model& model, const std::vector<cv::Point3f>& points,
            const std::vector<Feature>& features);

    /**
    * matches features using knn search with k = 2 and returning the matched
    * indices only if distance ratio of first match and second match is
    * above a threshold
    */
    static void matchFeatures(const cv::Mat& features1, 
            const cv::Mat& features2, std::vector<cv::DMatch>& matches);

    /**
    * Calculates the transformation matrix to transform points1 to
    * points2. Uses PCL's estimateRigidTransformationSVD
    */
    static void calculateTransform(const std::vector<cv::Point3f>& points1,
        const std::vector<cv::Point3f>& points2, cv::Mat& transform);

};

}

    
#endif /* MODEL_BUILDER_H */

