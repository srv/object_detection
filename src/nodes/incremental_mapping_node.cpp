#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transform.h>

#include <vision_msgs/Features3D.h>

#include "object_detection/pcl_descriptor.h"
#include "object_detection/model3d_alignment.h"
#include "object_detection/model3d_fusion.h"

typedef pcl::PointXYZ PointType;
typedef object_detection::PclDescriptor DescriptorType;

typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointCloud<DescriptorType> DescriptorCloud;

typedef object_detection::Model3D<PointType, DescriptorType> Model;

class IncrementalMappingNode
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber features_sub_;

  ros::Publisher pose_pub_;
  ros::Publisher model_pub_;
  ros::Publisher inlier_pub_;

  //tf::TransformBroadcaster tf_broadcaster_;

  double scale_error_threshold_;
  double ransac_threshold_;

  Model::Ptr map_;

public:
  IncrementalMappingNode() : nh_private_("~"),
    map_(new Model())
  {
    init();
  }

  ~IncrementalMappingNode()
  {
  }

  void init()
  {
    features_sub_ = nh_.subscribe("features_3d", 1, &IncrementalMappingNode::runUpdate, this);
  }

  void runUpdate(const vision_msgs::Features3D::ConstPtr& features_3d)
  {
    /*
    assert(features_3d->features_left.descriptor_data.size() / features_3d->features_left.key_points.size() == 64);
    assert(features_3d->features_left.key_points.size() == features_3d->world_points.size());

    Model::Ptr current_model(new Model());
    for (size_t i = 0; i < features_3d->features_left.key_points.size(); ++i)
    {
      PointType point;
      point.x = features_3d->world_points[i].x;
      point.y = features_3d->world_points[i].y;
      point.z = features_3d->world_points[i].z;
      DescriptorType descriptor;
      std::copy(features_3d->features_left.descriptor_data.begin() + i * 64,
                features_3d->features_left.descriptor_data.begin() + (i + 1) * 64,
                descriptor.data);
      current_model->addNewPoint(point, descriptor);
    }

    ROS_INFO("Received model with %zu features.",
        current_model->getPointCloud()->points.size());

    if (map_->isEmpty())
    {
      map_ = current_model;
    }
    else
    {
      object_detection::Model3DAlignment<Model> alignment;
      alignment.setSource(current_model);
      alignment.setTarget(map_);
      Eigen::Matrix4f transformation;
      alignment.align(transformation);
      // transformation is current cloud to map cloud,
      // invert to have the transform from initial camera
      // to current camera
      Eigen::Matrix4f cam_transform = transformation.inverse();
      std::cout << "Transformation: " << std::endl;
      std::cout << transformation << std::endl;
      std::cout << "Cam transform: " << std::endl;
      std::cout << cam_transform << std::endl;

      // transform current point inplace
      current_model->transformPoints(cam_transform);
      */

      /*
      object_detection::Model3DFusion<Model> fusion;
      fusion.setSource(current_model);
      fusion.setTarget(map_);
      fusion.fuse();
      */
    /*
    }

    ROS_INFO("Map has %zu points.",
        map_->getPointCloud()->points.size());
        */
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "incremental_mapping");
  IncrementalMappingNode incremental_mapping;
  ros::spin();
  return 0;
}

