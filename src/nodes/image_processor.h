#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>

/**
 * Base class for ROS image processors
 */
class ImageProcessor
{
 public:
  ImageProcessor() :
    it_(nh_)
  {
    camera_sub_ = it_.subscribeCamera("image", 1, 
        &ImageProcessor::imageCallback, this);
  }

  virtual void imageCallback(
      const sensor_msgs::ImageConstPtr& image_msg, 
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg) = 0;

 private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber camera_sub_;
};

