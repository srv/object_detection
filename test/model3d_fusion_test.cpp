#include <gtest/gtest.h>

#include "object_detection/model3d.h"
#include "object_detection/model3d_fusion.h"
#include "object_detection/pcl_descriptor.h"

using namespace object_detection;

pcl::PointXYZ randomPoint()
{
  return pcl::PointXYZ(rand() * 3.0 / RAND_MAX, rand() * 3.0 / RAND_MAX, rand() * 3.0 / RAND_MAX);
}

PclDescriptor randomDescriptor()
{
  PclDescriptor descriptor;
  for (int i = 0; i < 64; ++i)
  {
    descriptor.data[i] = 1.0 * rand() / RAND_MAX;
  }
  return descriptor;
}

TEST(Model3DFusion, fusionTest)
{
  typedef Model3D<pcl::PointXYZ, PclDescriptor> Model;
  Model::Ptr model_source(new Model());
  Model::Ptr model_target(new Model());

  std::vector<PclDescriptor> descriptors;
  for (int j = 0; j < 3; ++j)
  {
    descriptors.push_back(randomDescriptor());
  }
 
  model_source->addNewPoint(pcl::PointXYZ(0, 1, 0), descriptors);
  model_source->addNewPoint(pcl::PointXYZ(1, 1, 0), descriptors);
  model_source->addNewPoint(pcl::PointXYZ(1, 2, 0), descriptors);

  model_target->addNewPoint(pcl::PointXYZ(2, 1, 0), descriptors);
  model_target->addNewPoint(pcl::PointXYZ(1, 1, 0), descriptors);
  model_target->addNewPoint(pcl::PointXYZ(1, 2, 0), descriptors);

  Model3DFusion<Model> fusion;
  fusion.setSource(model_source);
  fusion.setTarget(model_target);
  fusion.fuse();

  EXPECT_EQ(model_target->getPointCloud()->points.size(), 4);
  EXPECT_EQ(model_target->getDescriptorCloud()->points.size(), 18);

  // check model consistency
  for (size_t i = 0; i < model_target->getPointCloud()->points.size(); ++i)
  {
    std::vector<int> descriptor_indices = model_target->getDescriptorIndicesForPointIndex(i);
    for (size_t j = 0; j < descriptor_indices.size(); ++j)
    {
      EXPECT_EQ(i, model_target->getPointIndexForDescriptorIndex(descriptor_indices[j]));
    }
  }
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

