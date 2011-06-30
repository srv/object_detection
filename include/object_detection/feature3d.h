#ifndef FEATURE3D_H
#define FEATURE3D_H

#include <opencv2/core/core.hpp>
#include "feature.h"

namespace object_detection {

struct Feature3D
{
    int world_point_index;
    std::vector<int> feature_indices;
    cv::Vec3b color;
};

}

#endif /* FEATURE3D_H */

