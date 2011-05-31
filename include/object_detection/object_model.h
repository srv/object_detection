#ifndef OBJECT_MODEL_H
#define OBJECT_MODEL_H

#include <list>

#include <cv.h>

#include "feature.h"

namespace object_detection {

struct Feature3D
{
    cv::Point3d world_point;
    std::vector<Feature> features;
    cv::Scalar color;
};

/**
 * \class ObjectModel
 * \author Stephan Wirth
 * \brief Data structure that defines an object model
 * The model is basically a 3d point set with feature descriptors attached.
 */
class ObjectModel
{

  public:

    /**
    * Creates an empty model
    */
    ObjectModel();

    void addFeature3D(const cv::Point3d& world_point, 
            const std::vector<Feature>& features, const cv::Scalar& color);

    /**
    * Renders all points with openGL commands
    */
    //void renderGL();

  private:

    std::vector<Feature3D> features_3d_;
};

}

#endif /* MASK_H */

