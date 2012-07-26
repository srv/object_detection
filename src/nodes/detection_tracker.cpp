#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <nav_msgs/Odometry.h>

#include "vision_msgs/Detection.h"

class DetectionTracker
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  message_filters::Subscriber<vision_msgs::Detection> sync_detection_sub_;
  message_filters::Subscriber<nav_msgs::Odometry> sync_odometry_sub_;
  message_filters::TimeSynchronizer<vision_msgs::Detection, nav_msgs::Odometry> synchronizer_;

  ros::Subscriber odometry_sub_;
  ros::Publisher detection_pub_;

  double timeout_;

  vision_msgs::DetectionConstPtr last_detection_;
  nav_msgs::OdometryConstPtr last_detection_odometry_;

  tf::TransformBroadcaster tf_broadcaster_;

public:
  DetectionTracker() :
    nh_private_("~"),
    sync_detection_sub_(nh_, "detection", 1),
    sync_odometry_sub_(nh_, "odometry", 1),
    synchronizer_(sync_detection_sub_, sync_odometry_sub_, 20)
  {
    synchronizer_.registerCallback(
        boost::bind(&DetectionTracker::detectionCallback, this, _1, _2));
    odometry_sub_ = nh_.subscribe(
        "odometry", 1, &DetectionTracker::odometryCallback, this);

    detection_pub_ = 
      nh_.advertise<vision_msgs::Detection>("tracked_detection", 1);

    // timeout for odometry based tracking of a detection
    nh_private_.param("timeout", timeout_, 5.0);
  }

  ~DetectionTracker()
  {
  }

  void detectionCallback(
    const vision_msgs::DetectionConstPtr& detection_msg,
    const nav_msgs::OdometryConstPtr& odometry_msg)
  {
    last_detection_ = detection_msg;
    last_detection_odometry_ = odometry_msg;
  }

  void odometryCallback(const nav_msgs::OdometryConstPtr& odometry_msg)
  {
    if (!last_detection_) return;
    if (last_detection_->header.stamp + ros::Duration(timeout_) > odometry_msg->header.stamp)
    {
      // detection still valid, compute current transform to detection
      tf::Pose current_pose;
      tf::poseMsgToTF(odometry_msg->pose.pose, current_pose);
      tf::Pose pose_at_detection;
      tf::poseMsgToTF(last_detection_odometry_->pose.pose, pose_at_detection);
      tf::Pose object_pose;
      tf::poseMsgToTF(last_detection_->pose.pose, object_pose);
      object_pose = current_pose.inverse() * pose_at_detection * object_pose;
      // create new detection msg with updated pose
      vision_msgs::Detection detection_msg = *last_detection_;
      detection_msg.header.stamp = odometry_msg->header.stamp;
      tf::poseTFToMsg(object_pose, detection_msg.pose.pose);

      detection_pub_.publish(detection_msg);

      // publish tf
      tf::StampedTransform stamped_transform(
          object_pose, detection_msg.header.stamp, 
          detection_msg.header.frame_id, "/target_tracked");
      tf_broadcaster_.sendTransform(stamped_transform);
    }
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "detection_tracker");
  DetectionTracker tracker;
  ros::spin();
  return 0;
}

