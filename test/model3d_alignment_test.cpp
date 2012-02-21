#include <gtest/gtest.h>

#include "object_detection/model3d.h"
#include "object_detection/model3d_alignment.h"
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

TEST(Model3DAlignment, emptyTest)
{  
  typedef Model3D<pcl::PointXYZ, PclDescriptor> Model;
  Model::Ptr model_source(new Model());
  Model::Ptr model_target(new Model());
  Model3DAlignment<Model> alignment;
  EXPECT_THROW(alignment.setSource(model_source), std::runtime_error);
  EXPECT_THROW(alignment.setTarget(model_target), std::runtime_error);
  Eigen::Matrix4f transformation_matrix;
  EXPECT_THROW(alignment.align(transformation_matrix), std::runtime_error);
}

TEST(Model3DAlignment, aligmentTest)
{
  typedef Model3D<pcl::PointXYZ, PclDescriptor> Model;
  Model::Ptr model_source(new Model());
  Model::Ptr model_target(new Model());

  for (int i = 0; i < 100; ++i)
  {
    std::vector<PclDescriptor> descriptors;
    int num_descriptors = rand() % 5;
    for (int j = 0; j < num_descriptors; ++j)
    {
      descriptors.push_back(randomDescriptor());
    }
    pcl::PointXYZ p = randomPoint();
    model_source->addNewPoint(p, descriptors);
    if (i < 50)
    {
      p.x += 0.1;
      p.y += 0.2;
      p.z += 0.3;
      model_target->addNewPoint(p, descriptors);
    }
  }

  std::cout << "Source model has " << model_source->getPointCloud()->size() 
            << " points with " << model_source->getDescriptorCloud()->size()
            << " descriptors." << std::endl;

  std::cout << "Target model has " << model_target->getPointCloud()->size() 
            << " points with " << model_target->getDescriptorCloud()->size()
            << " descriptors." << std::endl;

  Model3DAlignment<Model> alignment;
  alignment.setSource(model_source);
  alignment.setTarget(model_target);
  Eigen::Matrix4f transformation_matrix;

  pcl::registration::Correspondences correspondences;
  alignment.determineCorrespondences(correspondences);
  EXPECT_LE(correspondences.size(), 50);

  alignment.align(transformation_matrix);

  std::cout << "Transformation: \n" << transformation_matrix << std::endl;

  EXPECT_NEAR(transformation_matrix(0, 0), 1.0, 0.001);
  EXPECT_NEAR(transformation_matrix(1, 0), 0.0, 0.001);
  EXPECT_NEAR(transformation_matrix(2, 0), 0.0, 0.001);
  EXPECT_NEAR(transformation_matrix(3, 0), 0.0, 0.001);
  
  EXPECT_NEAR(transformation_matrix(0, 1), 0.0, 0.001);
  EXPECT_NEAR(transformation_matrix(1, 1), 1.0, 0.001);
  EXPECT_NEAR(transformation_matrix(2, 1), 0.0, 0.001);
  EXPECT_NEAR(transformation_matrix(3, 1), 0.0, 0.001);

  EXPECT_NEAR(transformation_matrix(0, 2), 0.0, 0.001);
  EXPECT_NEAR(transformation_matrix(1, 2), 0.0, 0.001);
  EXPECT_NEAR(transformation_matrix(2, 2), 1.0, 0.001);
  EXPECT_NEAR(transformation_matrix(3, 2), 0.0, 0.001);

  EXPECT_NEAR(transformation_matrix(0, 3), 0.1, 0.001);
  EXPECT_NEAR(transformation_matrix(1, 3), 0.2, 0.001);
  EXPECT_NEAR(transformation_matrix(2, 3), 0.3, 0.001);
  EXPECT_NEAR(transformation_matrix(3, 3), 1.0, 0.001);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

