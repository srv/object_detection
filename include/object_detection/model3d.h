#ifndef MODEL3D_H
#define MODEL3D_H

#include <pcl/point_cloud.h>

namespace object_detection {

/**
 * \class Model3D
 * \author Stephan Wirth
 * \brief Data structure that defines a 3D model
 * The model is basically a 3d point set with feature descriptors attached.
 */
template <typename PointT, typename DescriptorT>
class Model3D
{

public:

  typedef PointT PointType;
  typedef DescriptorT DescriptorType;
  typedef typename pcl::PointCloud<PointT> PointCloud;
  typedef typename PointCloud::Ptr PointCloudPtr;
  typedef typename PointCloud::ConstPtr PointCloudConstPtr;
  typedef typename pcl::PointCloud<DescriptorT> DescriptorCloud;
  typedef typename DescriptorCloud::Ptr DescriptorCloudPtr;
  typedef typename DescriptorCloud::ConstPtr DescriptorCloudConstPtr;

  typedef std::map<int, int> IndexToIndexMap;
  typedef boost::shared_ptr<IndexToIndexMap>  IndexToIndexMapPtr;
  typedef boost::shared_ptr<const IndexToIndexMap>  IndexToIndexMapConstPtr;

  /**
   * Creates an empty model
   */
  Model3D()
  {
    point_cloud_.reset(new PointCloud());
    descriptor_cloud_.reset(new DescriptorCloud());
    descriptor_to_world_point_.reset(new IndexToIndexMap());
  }

  inline PointCloudConstPtr getPointCloud() const { return point_cloud_; }

  inline DescriptorCloudConstPtr getDescriptorCloud() const { return descriptor_cloud_; }

  int getPointIndexForDescriptorIndex(int descriptor_index) const;
    
  void attachDescriptor(int world_point_index, const DescriptorT& descriptor);

  void addNewPoint(const PointT& world_point, const std::vector<DescriptorT>& descriptors);

  typedef boost::shared_ptr<Model3D> Ptr;
  typedef boost::shared_ptr<const Model3D> ConstPtr;

private:

  PointCloudPtr point_cloud_;
  DescriptorCloudPtr descriptor_cloud_;

  IndexToIndexMapPtr descriptor_to_world_point_;

  std::map<int, std::vector<int> > world_point_to_descriptors_;

};

}

#include "object_detection/impl/model3d.hpp"

#endif /* MODEL3D_H */

