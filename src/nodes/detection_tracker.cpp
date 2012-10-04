#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <nav_msgs/Odometry.h>

#include <vision_msgs/DetectionArray.h>

class DetectionTracker
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  message_filters::Subscriber<vision_msgs::DetectionArray> sync_detection_sub_;
  message_filters::Subscriber<nav_msgs::Odometry> sync_odometry_sub_;
  message_filters::TimeSynchronizer<vision_msgs::DetectionArray, nav_msgs::Odometry> synchronizer_;

  ros::Subscriber odometry_sub_;
  ros::Publisher detections_pub_;

  double timeout_;

  vision_msgs::DetectionArrayConstPtr last_detections_;
  nav_msgs::OdometryConstPtr last_detections_odometry_;

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

    detections_pub_ = 
      nh_.advertise<vision_msgs::DetectionArray>("tracked_detections", 1);

    // timeout for odometry based tracking of a detection
    nh_private_.param("timeout", timeout_, 5.0);
  }

  ~DetectionTracker()
  {
  }

  void detectionCallback(
    const vision_msgs::DetectionArrayConstPtr& detections_msg,
    const nav_msgs::OdometryConstPtr& odometry_msg)
  {
    last_detections_ = detections_msg;
    last_detections_odometry_ = odometry_msg;
  }

  void odometryCallback(const nav_msgs::OdometryConstPtr& odometry_msg)
  {
    if (!last_detections_) return;
    if (last_detections_->header.stamp + ros::Duration(timeout_) > odometry_msg->header.stamp)
    {
      // detection still valid, compute current transform to detection
      tf::Pose current_pose;
      tf::poseMsgToTF(odometry_msg->pose.pose, current_pose);
      tf::Pose pose_at_detection;
      tf::poseMsgToTF(last_detections_odometry_->pose.pose, pose_at_detection);
      vision_msgs::DetectionArray tracked_detections_msg;
      for (size_t i = 0; i < last_detections_->detections.size(); ++i)
      {
        tf::Pose training_pose;
        tf::poseMsgToTF(last_detections_->detections[i].training_pose, training_pose);
        training_pose = current_pose.inverse() * pose_at_detection * training_pose;
        // create new detection msg with updated pose
        vision_msgs::Detection detection_msg = last_detections_->detections[i];
        detection_msg.header.stamp = odometry_msg->header.stamp;
        tf::poseTFToMsg(training_pose, detection_msg.training_pose);
        tracked_detections_msg.detections.push_back(detection_msg);
      }
      detections_pub_.publish(tracked_detections_msg);

      // publish tf
      /*
      tf::StampedTransform stamped_transform(
          object_pose, detection_msg.header.stamp, 
          detection_msg.header.frame_id, "/target_tracked");
      tf_broadcaster_.sendTransform(stamped_transform);
      */
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

