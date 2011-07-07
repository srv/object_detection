#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <vector>

namespace cv {
    class Mat;
    class DMatch;
    template <class T> class Point3_;
    typedef Point3_<float> Point3f;
}

namespace object_detection {

class Feature;
class Model;

/**
 * \class Registration
 * \author Stephan Wirth
 * \brief Matching of 3D features
 */
class Registration
{

  public:

    void estimateTransformation(const std::vector<cv::Point3f>& points1,
            const std::vector<Feature>& features1, 
            const std::vector<cv::Point3f>& points2,
            const std::vector<Feature>& features2,
            cv::Mat& transformation,
            std::vector<DMatch>& inlier_matches);


    /**
    * Extend the given model by matching given features and adding
    * those features/points that are new
    */
    //static void extend(Model& model, const std::vector<cv::Point3f>& points,
    //        const std::vector<Feature>& features);

    /**
    * matches features using knn search with k = 2 and returning the matched
    * indices only if distance ratio of first match and second match is
    * above a threshold
    */
    //static void matchFeatures(const cv::Mat& features1, 
    //        const cv::Mat& features2, std::vector<cv::DMatch>& matches);

};

}

    
#endif /* MODEL_BUILDER_H */

