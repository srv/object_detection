#ifndef MODEL_3D_ALIGNMENT_H
#define MODEL_3D_ALIGNMENT_H

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/correspondence_types.h>

#include "object_detection/model3d.h"

namespace object_detection
{

template <typename ModelT>
class Model3DAlignment
{
public:

  typedef typename ModelT::PointCloud PointCloud;
  typedef typename PointCloud::Ptr PointCloudPtr;
  typedef typename PointCloud::ConstPtr PointCloudConstPtr;

  typedef typename ModelT::DescriptorCloud DescriptorCloud;
  typedef typename DescriptorCloud::ConstPtr DescriptorCloudConstPtr;
  typedef typename ModelT::DescriptorType DescriptorType;

  typedef typename ModelT::ConstPtr ModelConstPtr;
  
  typedef typename pcl::KdTreeFLANN<DescriptorType> DescriptorKdTree;
  typedef typename DescriptorKdTree::Ptr DescriptorKdTreePtr;

  Model3DAlignment()
  {
    target_descriptor_tree_.reset(new DescriptorKdTree());
  }

  void setSource(const ModelConstPtr& source_model);
  void setTarget(const ModelConstPtr& target_model);

  void align(Eigen::Matrix4f& transformation_matrix);

  void determineCorrespondences(pcl::registration::Correspondences& correspondences);

  void filterCorrespondences(
      const pcl::registration::CorrespondencesConstPtr& original_correspondences,
      pcl::registration::Correspondences& filtered_correspondences);

private:

  ModelConstPtr source_model_;
  ModelConstPtr target_model_;

  DescriptorKdTreePtr target_descriptor_tree_;

};

} // end of namespace

#include "object_detection/impl/model3d_alignment.hpp"

#endif

