#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>

template <typename ModelT>
void object_detection::Model3DAlignment<ModelT>::setSource(const ModelConstPtr& source_model)
{
  source_model_ = source_model;
}

template <typename ModelT>
void object_detection::Model3DAlignment<ModelT>::setTarget(const ModelConstPtr& target_model)
{
  target_model_ = target_model;
  target_descriptor_tree_->setInputCloud(target_model_->getDescriptorCloud());
}

template <typename ModelT>
void object_detection::Model3DAlignment<ModelT>::align(Eigen::Matrix4f& transformation_matrix)
{
  pcl::registration::CorrespondencesPtr correspondences(new pcl::registration::Correspondences());

  determineCorrespondences(*correspondences);

  std::cout << "found " << correspondences->size() << " correspondences." << std::endl;

  // filter correspondences
  typedef typename PointCloud::PointType PointType;
  pcl::registration::CorrespondenceRejectorSampleConsensus<PointType> rejector;
  rejector.setInputCloud(source_model_->getPointCloud());
  rejector.setTargetCloud(target_model_->getPointCloud());
  rejector.setMaxIterations(1000); //TODO setter
  rejector.setInlierThreshold(0.05); // TODO setter
  rejector.setInputCorrespondences(correspondences);

  pcl::registration::Correspondences filtered_correspondences;
  rejector.getCorrespondeces(filtered_correspondences);

  // determine rigid transformation
  pcl::registration::TransformationEstimationSVD<PointType, PointType> transformation_estimator;
  transformation_estimator.estimateRigidTransformation(
      *(source_model_->getPointCloud()),
      *(target_model_->getPointCloud()),
      filtered_correspondences,
      transformation_matrix);
}

template <typename ModelT>
void object_detection::Model3DAlignment<ModelT>::determineCorrespondences(pcl::registration::Correspondences& correspondences)
{
  assert(source_model_ != 0);
  assert(target_model_ != 0);

  DescriptorCloudConstPtr source_descriptors = source_model_->getDescriptorCloud();

  std::vector<int> target_indices(source_model_->getPointCloud()->size(), -1);
  std::vector<float> target_distances(source_model_->getPointCloud()->size(), 
      std::numeric_limits<float>::max());

  for (size_t sd_index = 0; sd_index < source_descriptors->size(); ++sd_index)
  {
    std::vector<int> k_indices(2);
    std::vector<float> k_distances(2);
    int num_neighbors_found = 
      target_descriptor_tree_->nearestKSearch(
          (*source_descriptors), sd_index, 2, k_indices, k_distances);
    bool match_found = false;
    if (num_neighbors_found == 1)
    {
      match_found = true;
    }
    else if (num_neighbors_found == 2)
    {
      if (k_indices[0] == k_indices[1])
      {
        match_found = true;
      }
      else if (k_distances[0] / k_distances[1] < 0.8) // TODO setter for threshold
      {
        match_found = true;
      }
    }

    if (match_found)
    {
      int point_index_query = source_model_->getPointIndexForDescriptorIndex(sd_index);
      int point_index_match = target_model_->getPointIndexForDescriptorIndex(k_indices[0]);
      // check if match is better than current
      if (k_distances[0] < target_distances[point_index_query])
      {
        target_indices[point_index_query] = point_index_match;
        target_distances[point_index_query] = k_distances[0];
      }
    }
  }

  correspondences.clear();
  for (size_t i = 0; i < target_indices.size(); ++i)
  {
    if (target_indices[i] >= 0)
    {
      pcl::registration::Correspondence correspondence;
      correspondence.indexQuery = i;
      correspondence.indexMatch = target_indices[i];
      correspondence.distance = target_distances[i];
      correspondences.push_back(correspondence);
    }
  }
}

