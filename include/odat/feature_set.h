#ifndef FEATURE_SET_H_
#define FEATURE_SET_H_

#include <opencv2/features2d/features2d.hpp>

namespace odat
{
  struct FeatureSet
  {
    // key points (feature locations)
    std::vector<cv::KeyPoint> key_points;

    // name of the descriptor (e.g. "SURF", "CvSIFT", etc.)
    std::string descriptor_name;

    // descriptors, one by row, number of rows == number of key points
    cv::Mat descriptors;
  };
}

#endif

