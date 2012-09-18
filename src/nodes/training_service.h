#include <ros/ros.h>
#include <vision_msgs/TrainDetector.h>

/**
 * Base class for ROS services for detector training
 */
class TrainingService
{
 public:
  TrainingService() : nh_("~")
  {
    service_ = nh_.advertiseService("train", &TrainingService::train, this);
  }

  virtual bool train(
      vision_msgs::TrainDetector::Request& training_request,
      vision_msgs::TrainDetector::Response& training_response) = 0;

 private:
  ros::NodeHandle nh_;
  ros::ServiceServer service_;
};

