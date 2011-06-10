#ifndef FEATURE3D_H
#define FEATURE3D_H

#include <opencv2/core/core.hpp>
#include "feature.h"

namespace object_detection {

struct Feature3D
{
    cv::Point3d world_point;
    std::vector<Feature> features;
    cv::Vec3b color;
};

}

#endif /* FEATURE3D_H */

