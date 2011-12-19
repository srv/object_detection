
#include "object_detection/model3d.h"
#include <iostream>

using namespace object_detection;

Model3D::Model3D()
{
}

cv::Mat Model3D::getDescriptors() const
{
  return descriptors_;
}

cv::Point3d Model3D::getWorldPointForDescriptor(int descriptor_index) const
{
  std::map<int, int>::const_iterator iter =
    descriptor_to_world_point_.find(descriptor_index);
  assert(iter != descriptor_to_world_point_.end());
  int world_point_index = (*iter).second;
  assert(world_point_index >= 0 && world_point_index < (int)world_points_.size());
  return world_points_[world_point_index];
}


void Model3D::attachFeature(int world_point_index, const cv::KeyPoint& key_point,
                            const cv::Mat& descriptor)
{
  assert(world_point_index >= 0 && world_point_index < (int)world_points_.size());

  key_points_.push_back(key_point);
  descriptors_.resize(descriptors_.rows + 1);
  int descriptor_index = descriptors_.rows - 1;
  cv::Mat target = descriptors_.row(descriptor_index);
  descriptor.copyTo(target);
  world_point_to_descriptors_[world_point_index].push_back(descriptor_index);
  descriptor_to_world_point_[descriptor_index] = world_point_index;

}

void Model3D::addFeature(const cv::KeyPoint& key_point,
                         const cv::Mat& descriptor,
                         const cv::Point3d& world_point)
{
  world_points_.push_back(world_point);
  int world_point_index = world_points_.size() - 1;
  attachFeature(world_point_index, key_point, descriptor);
}

std::ostream& object_detection::operator<<(std::ostream& ostr, const Model3D& model)
{
  ostr << "Model with " << model.world_points_.size() << " world points, ";
  ostr << model.key_points_.size() << " features.";

  /*
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
  */
  return ostr;
}

