#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <vector>

namespace cv
{
    class Mat;
    class DMatch;
}

namespace object_detection {

/**
 * \class FeatureMatcher
 * \author Stephan Wirth
 * \brief Matches two sets of features
 */
class FeatureMatcher
{

  public:
    /**
    * matches features using knn search with k = 2 and returning the matched
    * indices only if distance ratio of first match and second match is
    * above a threshold
    */
    static void matchFeatures(const cv::Mat& features1, 
            const cv::Mat& features2, std::vector<cv::DMatch>& matches);
};

}

    
#endif /* FEATURE_MATCHER_H */

