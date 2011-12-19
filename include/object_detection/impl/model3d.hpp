
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
std::vector<int> object_detection::Model3D<PointT, DescriptorT>::getDescriptorIndicesForPointIndex(int point_index) const
{
  IndexToIndicesMap::const_iterator iter =
    world_point_to_descriptors_->find(point_index);
  if (iter == world_point_to_descriptors_->end())
  {
    throw std::runtime_error("InvalidPointIndex");
  }
  return iter->second;
}

template <typename PointT, typename DescriptorT>
void object_detection::Model3D<PointT, DescriptorT>::attachDescriptor(int world_point_index, const DescriptorT& descriptor)
{
  assert(world_point_index >= 0 && world_point_index < point_cloud_->points.size());
  descriptor_cloud_->push_back(descriptor);
  int descriptor_index = descriptor_cloud_->points.size() - 1;
  (*descriptor_to_world_point_)[descriptor_index] = world_point_index;
  (*world_point_to_descriptors_)[world_point_index].push_back(descriptor_index);
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
    (*world_point_to_descriptors_)[point_index].push_back(descriptor_index);
  }
}

