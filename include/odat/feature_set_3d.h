#ifndef FEATURE_SET_3D_H_
#define FEATURE_SET_3D_H_

#include "odat/feature_set.h"

namespace odat
{
  /**
  * A stereo feature set bundles features together with the
  * corresponding estimated 3d points.
  */
  struct FeatureSet3D
  {
    // features from left image
    FeatureSet features_left;

    // the estimated world points in camera coordinates,
    // seen from the left camera
    std::vector<cv::Point3f> world_points;

    // mapping from feature descriptor index to world point index
    std::vector<unsigned int> descriptor_index_to_world_point_index;

  };
}

#endif

