#ifndef MODEL_3D_FUSION_H
#define MODEL_3D_FUSION_H

#include <pcl/kdtree/kdtree_flann.h>

#include "object_detection/model3d.h"

namespace object_detection
{

template <typename ModelT>
class Model3DFusion
{
public:

  typedef typename ModelT::PointCloud PointCloud;
  typedef typename PointCloud::Ptr PointCloudPtr;
  typedef typename PointCloud::ConstPtr PointCloudConstPtr;

  typedef typename ModelT::DescriptorCloud DescriptorCloud;
  typedef typename DescriptorCloud::ConstPtr DescriptorCloudConstPtr;
  typedef typename ModelT::DescriptorType DescriptorType;
  typedef typename ModelT::PointType PointType;

  typedef typename ModelT::Ptr ModelPtr;
  typedef typename ModelT::ConstPtr ModelConstPtr;
  
  typedef typename pcl::KdTreeFLANN<PointType> PointKdTree;
  typedef typename PointKdTree::Ptr PointKdTreePtr;

  Model3DFusion()
  {
    target_tree_.reset(new PointKdTree());
  }

  void setSource(const ModelConstPtr& source_model) { 
    source_model_ = source_model; 
  }

  // set the target model, will be modified by fuse()
  void setTarget(const ModelPtr& target_model) {
    target_model_ = target_model;
  }

  // integrates the source model into the target model
  void fuse();

private:

  ModelConstPtr source_model_;
  ModelPtr target_model_;

  PointKdTreePtr target_tree_;

};

} // end of namespace

#include "object_detection/impl/model3d_fusion.hpp"

#endif

