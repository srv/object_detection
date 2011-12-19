
template <typename PointT, typename DescriptorT>
int object_detection::Model3D<PointT, DescriptorT>::getPointIndexForDescriptorIndex(int descriptor_index) const
{
  IndexToIndexMap::const_iterator iter =
    descriptor_to_world_point_->find(descriptor_index);
  if (iter == descriptor_to_world_point_->end())
  {
    throw std::runtime_error("InvalidDescriptorIndex");
  }
  return iter->second;
}

template <typename PointT, typename DescriptorT>
void object_detection::Model3D<PointT, DescriptorT>::addNewPoint(const PointT& world_point, const std::vector<DescriptorT>& descriptors)
{
  // todo locking?
  point_cloud_->push_back(world_point);
  int point_index = point_cloud_->points.size() - 1;

  typename std::vector<DescriptorT>::const_iterator iter;
  for (iter = descriptors.begin(); iter != descriptors.end(); ++iter)
  {
    descriptor_cloud_->push_back(*iter);
    int descriptor_index = descriptor_cloud_->points.size() - 1;
    (*descriptor_to_world_point_)[descriptor_index] = point_index;
  }
}

