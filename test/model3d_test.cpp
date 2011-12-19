#include <gtest/gtest.h>

#include "object_detection/model3d.h"
#include "object_detection/pcl_descriptor.h"

using namespace object_detection;

TEST(Model3D, integrityTest)
{
  typedef Model3D<pcl::PointXYZ, PclDescriptor> Model;
  Model model;
  Model::PointCloudConstPtr points = model.getPointCloud();
  Model::DescriptorCloudConstPtr descriptors = model.getDescriptorCloud();
  EXPECT_EQ(points->size(), 0);
  EXPECT_EQ(descriptors->size(), 0);

  for (int i = 0; i < 10; ++i)
  {
    std::vector<PclDescriptor> descriptors;
    for (int j = 0; j < 3; ++j)
    {
      descriptors.push_back(PclDescriptor());
    }
    model.addNewPoint(pcl::PointXYZ(), descriptors);
  }

  EXPECT_EQ(points->size(), 10);
  EXPECT_EQ(descriptors->size(), 30);

  EXPECT_THROW(model.getPointIndexForDescriptorIndex(-1), std::runtime_error);
  EXPECT_THROW(model.getPointIndexForDescriptorIndex(30), std::runtime_error);

  for (int i = 0; i < 30; ++i)
  {
    int point_index;
    EXPECT_NO_THROW({
        point_index = model.getPointIndexForDescriptorIndex(i);
        });
    EXPECT_EQ(point_index, i/3);
  }
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

