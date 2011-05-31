#ifndef OBJECT_MODEL_H
#define OBJECT_MODEL_H

#include <cv.h>

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


  private:

    std::vector<cv::Point3d> world_points_;
    std::vector<cv::Mat> descriptors_;

    std::vector<std::list<cv::Mat> > world_point_indices_to_descriptors_;
    std::vector<int> descriptor_index_to_world_point_index_;
};

}

#endif /* MASK_H */

