
template <typename ModelT>
void object_detection::Model3DFusion<ModelT>::fuse()
{
  target_tree_->setInputCloud(target_model_->getPointCloud());
  PointCloudConstPtr source_cloud = source_model_->getPointCloud();
  DescriptorCloudConstPtr source_descriptor_cloud = source_model_->getDescriptorCloud();
  for (size_t i = 0; i < source_cloud->size(); ++i)
  {
    std::vector<int> k_indices(1);
    std::vector<float> k_distances(1);
    int found_k = target_tree_->nearestKSearch(*source_cloud, i, 1, k_indices, k_distances);
    if (found_k == 1)
    {
      std::vector<int> descriptor_indices = source_model_->getDescriptorIndicesForPointIndex(i);
      if (k_distances[0] < 0.01) // TODO parameter for fusion threshold dependent on z?
      {
        for (size_t j = 0; j < descriptor_indices.size(); ++j)
        {
          target_model_->attachDescriptor(k_indices[0], source_descriptor_cloud->points[descriptor_indices[j]]);
        }
      }
      else
      {
        std::vector<typename ModelT::DescriptorType> descriptors(descriptor_indices.size());
        for (size_t j = 0; j < descriptor_indices.size(); ++j)
        {
          descriptors[j] = source_descriptor_cloud->points[descriptor_indices[j]];
        }
        target_model_->addNewPoint(source_cloud->points[i], descriptors);
      }
    }
  }
}

