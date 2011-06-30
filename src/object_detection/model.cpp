
#include "model.h"
#include <iostream>

using namespace object_detection;

Model::Model()
{
}

cv::Mat Model::getFeatureData() const
{
    cv::Mat feature_data;
    if (features_.size() == 0)
    {
        return feature_data;
    }
    feature_data.create(features_.size(), features_[0].descriptor.size(), CV_32F);

    for (size_t i = 0; i < features_.size(); ++i)
    {
        std::copy(features_[i].descriptor.begin(),
                  features_[i].descriptor.end(),
                  feature_data.ptr<float>(i));
    }
    return feature_data;
}

void Model::addFeature(const cv::Point3f& world_point, 
        const Feature& feature)
{
    world_points_.push_back(world_point);
    int world_point_index = world_points_.size() - 1;
    features_.push_back(feature);
    int feature_index = features_.size() - 1;
    feature_index_to_world_point_index_[feature_index] = world_point_index;
}

cv::Point3f Model::getWorldPoint(int feature_index) const
{
    std::map<int, int>::const_iterator iter = 
        feature_index_to_world_point_index_.find(feature_index);
    assert(iter != feature_index_to_world_point_index_.end());
    int world_point_index = (*iter).second;
    assert(world_point_index >= 0 && world_point_index < (int)world_points_.size());
    return world_points_[world_point_index];
}

std::ostream& object_detection::operator<<(std::ostream& ostr, const Model& model)
{
    ostr << model.world_points_.size() << " world points, ";
    ostr << model.features_.size() << " features." << std::endl;
    ostr << "world points:" << std::endl;
    for (size_t i = 0; i < model.world_points_.size(); ++i)
    {
        ostr << "(" << i << "):" << model.world_points_[i] << " ";
    }
    ostr << std::endl;
    ostr << "features:" << std::endl;
    for (size_t i = 0; i < model.world_points_.size(); ++i)
    {
        ostr << "(" << i << "):";
        for (size_t j = 0; j < model.features_[i].descriptor.size(); ++j)
        {
            ostr << " " << model.features_[i].descriptor[j];
        }
        ostr << std::endl;
    }
    return ostr;
}
