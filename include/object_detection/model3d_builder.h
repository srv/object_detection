#ifndef MODEL3D_BUILDER_H
#define MODEL3D_BUILDER_H

#include "object_detection/model3d.h"
#include "odat/feature_set_3d.h"

#include <opencv2/features2d/features2d.hpp>

namespace object_detection {

/**
 * \class Model3DBuilder
 * \brief Incrementally updates a Model3D
 */
class Model3DBuilder
{

public:

  Model3DBuilder() {}

  void setModel(const Model3D::Ptr& model) { model_ = model; }

  /**
  * \brief incorporates new features into the managed model
  * \return true on success, false if features could not be added
  */
  bool update(const odat::FeatureSet3D& features);

private:

  Model3D::Ptr model_;

};

}


#endif /* MODEL3D_BUILDER_H */

