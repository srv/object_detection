#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transform.h>

#include <vision_msgs/Features3D.h>
#include <nav_msgs/Odometry.h>

#include <tf/transform_broadcaster.h>

#include "object_detection/pcl_descriptor.h"
#include "object_detection/model3d_alignment.h"
#include "object_detection/model3d_fusion.h"

typedef pcl::PointXYZ PointType;
typedef object_detection::PclDescriptor DescriptorType;

typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointCloud<DescriptorType> DescriptorCloud;

typedef object_detection::Model3D<PointType, DescriptorType> Model;

class Odometer3DNode
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber features_sub_;

  ros::Publisher odometry_pub_;

  tf::TransformBroadcaster tf_broadcaster_;

  double scale_error_threshold_;
  double ransac_threshold_;

  Model::Ptr previous_model_;

  Eigen::Matrix4f integrated_transform_;

public:
  Odometer3DNode() : nh_private_("~")
  {
    init();
  }

  ~Odometer3DNode()
  {
  }

  void init()
  {
    features_sub_ = nh_private_.subscribe("features_3d", 1, &Odometer3DNode::runUpdate, this);
    integrated_transform_.setIdentity();

    odometry_pub_ = nh_private_.advertise<nav_msgs::Odometry>("odometry", 1);
  }

  void runUpdate(const vision_msgs::Features3D::ConstPtr& features_3d)
  {
    assert(features_3d->features_left.descriptor_data.cols == 64);
    assert(features_3d->features_left.key_points.size() == features_3d->world_points.size());

    Model::Ptr current_model(new Model());
    const float* data_start = reinterpret_cast<const float*>(&(features_3d->features_left.descriptor_data.data[0]));
    for (size_t i = 0; i < features_3d->features_left.key_points.size(); ++i)
    {
      PointType point;
      point.x = features_3d->world_points[i].x;
      point.y = features_3d->world_points[i].y;
      point.z = features_3d->world_points[i].z;
      DescriptorType descriptor;
      std::copy(data_start + i * 64, data_start + (i + 1) * 64, descriptor.data);
      current_model->addNewPoint(point, descriptor);
    }

    ROS_INFO("Received model with %zu features.",
        current_model->getPointCloud()->points.size());

    if (previous_model_)
    {
      object_detection::Model3DAlignment<Model> alignment;
      alignment.setSource(current_model);
      alignment.setTarget(previous_model_);
      Eigen::Matrix4f transformation;
      alignment.align(transformation);
      std::cout << "Transformation: " << std::endl;
      std::cout << transformation << std::endl;
      integrated_transform_ *= transformation; // * integrated_transform_;
    }
    previous_model_ = current_model;
    publishOdometry(features_3d->header);
  }

  void publishOdometry(const std_msgs::Header& header)
  {
    std::string base_frame_id = header.frame_id + "_initial_pose";
    nav_msgs::Odometry odometry_msg;
    odometry_msg.header = header;
    odometry_msg.header.frame_id = base_frame_id;
    odometry_msg.child_frame_id = header.frame_id;
    float x = integrated_transform_(0, 3);
    float y = integrated_transform_(1, 3);
    float z = integrated_transform_(2, 3);
    odometry_msg.pose.pose.position.x = x;
    odometry_msg.pose.pose.position.y = y;
    odometry_msg.pose.pose.position.z = z;

    odometry_pub_.publish(odometry_msg);

    tf::Transform transform(
        btMatrix3x3(
          integrated_transform_(0, 0),
          integrated_transform_(0, 1),
          integrated_transform_(0, 2),
          integrated_transform_(1, 0),
          integrated_transform_(1, 1),
          integrated_transform_(1, 2),
          integrated_transform_(2, 0),
          integrated_transform_(2, 1),
          integrated_transform_(2, 2)),
        btVector3(x, y, z));
    tf_broadcaster_.sendTransform(
        tf::StampedTransform(transform, header.stamp,
          base_frame_id, header.frame_id));
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "odometer3d_node");
  Odometer3DNode node;
  ros::spin();
  return 0;
}

