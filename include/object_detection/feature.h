#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/features2d/features2d.hpp>

namespace object_detection
{

/**
* Struct to hold a feature
*/
struct Feature
{
    cv::KeyPoint key_point;
    std::vector<float> descriptor;
};

} // end of namespace object_detection

#endif // defined FEATURE_H


