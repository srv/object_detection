#ifndef OBJECT_MODEL_H
#define OBJECT_MODEL_H

#include <vector>
#include "feature3d.h"

namespace object_detection {

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

    void addFeature3D(const Feature3D& feature3d);

    /**
    * Renders all points with openGL commands
    */
    //void renderGL();

  private:

    std::vector<Feature3D> features3d_;
};

}

#endif /* OBJECT_MODEL_H */

