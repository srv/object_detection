#include <gtest/gtest.h>

#include <pcl/kdtree/kdtree_flann.h>
#include "object_detection/pcl_descriptor.h"

using namespace object_detection;

float randDescriptorValue()
{
  return 200.0 * rand() / RAND_MAX;
}

TEST(PclDescriptor, distanceTest)
{
  srand(time(0));
  PclDescriptor descriptor1;

  int descriptor_size = sizeof(descriptor1.data) / sizeof(descriptor1.data[0]);
  std::cout << "Descriptor size: " << descriptor_size << std::endl;
  std::generate(descriptor1.data, descriptor1.data + descriptor_size, randDescriptorValue);
  std::cout << descriptor1 << std::endl;

  PclDescriptor descriptor2;
  descriptor2 = descriptor1;

  for (int i = 0; i < descriptor_size; ++i)
  {
    EXPECT_DOUBLE_EQ(descriptor2.data[i], descriptor1.data[i]);
  }

  for (int i = 0; i < descriptor_size; ++i)
  {
    PclDescriptor cmp = descriptor1;
    cmp.data[i] += 2;
    pcl::KdTreeFLANN<PclDescriptor> kd_tree;
    pcl::PointCloud<PclDescriptor>::Ptr cloud(new pcl::PointCloud<PclDescriptor>());
    cloud->push_back(descriptor1);
    kd_tree.setInputCloud(cloud);
    std::vector<int> k_indices(1);
    std::vector<float> k_distances(1);
    kd_tree.nearestKSearch(cmp, 1, k_indices, k_distances);
    EXPECT_NEAR(k_distances[0], 4.0, 0.0001);
  }
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

