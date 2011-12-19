#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <map>
#include <ostream>

#include "feature.h"

namespace object_detection {

/**
 * \class Model
 * \author Stephan Wirth
 * \brief Data structure that defines a 3D model
 * The model is basically a 3d point set with feature descriptors attached.
 */
class Model
{

  public:

    /**
    * Creates an empty model
    */
    Model();

    /**
    * \return all features in a floating point matrix, each row has
    *         descriptor data from one feature.
    */
    cv::Mat getFeatureData() const;

    /**
    * Add a new feature
    */
    void addFeature(const cv::Point3f& world_point, const Feature& feature);

    /**
    * \feature_index the index of the feature for which the world point
    *                is requested. Must be valid.
    * \return the world point for the feature at given index
    */
    cv::Point3f getWorldPoint(int feature_index) const;

    friend std::ostream& operator<<(std::ostream& ostr, const Model& model);

    void writeToPCD(const std::string& file_name);

  private:

    std::vector<cv::Point3f> world_points_;
    std::vector<Feature> features_;
    std::map<int, int> feature_index_to_world_point_index_;

};

std::ostream& operator<<(std::ostream& ostr, const object_detection::Model& model);
}

    
#endif /* MODEL_H */

