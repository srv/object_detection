#include "object_detection/model3d_builder.h"
#include "object_detection/alignment.h"
#include "object_detection/pcl_descriptor.h"

bool object_detection::Model3DBuilder::update(
    const odat::FeatureSet3D& features_3d)
{
  assert(model_.get() != NULL);
  
  typedef pcl::PointXYZ PointType;
  typedef object_detection::PclDescriptor FeatureType;

  typedef object_detection::Alignment<PointType, PointType, FeatureType> AlignmentType;

  AlignmentType alignment;
  pcl::PointCloud<PointType> target_cloud;
  pcl::PointCloud<PointType> source_cloud;

  std::map<int, int> feature_index_to_point_index;


  return false;
}

