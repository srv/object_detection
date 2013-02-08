#include <vision_msgs/DetectionArray.h>

#include "detector.h"
#include "training_service.h"
#include "stereo_processor.h"

class StereoDetector :
  public Detector, 
  public TrainingService,
  public object_detection::StereoImageProcessor
{
 public:
  StereoDetector() : Detector(), TrainingService(), object_detection::StereoImageProcessor("raw"), nh_("~")
  {
    detections_pub_ = nh_.advertise<vision_msgs::DetectionArray>("detections", 1);
  }

  virtual void detect(
     const sensor_msgs::ImageConstPtr& l_image_msg, 
     const sensor_msgs::ImageConstPtr& r_image_msg, 
     const sensor_msgs::CameraInfoConstPtr& l_camera_info_msg,
     const sensor_msgs::CameraInfoConstPtr& r_camera_info_msg,
     vision_msgs::DetectionArray& detections) = 0;

 private:

  virtual void stereoImageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg, 
      const sensor_msgs::ImageConstPtr& r_image_msg, 
      const sensor_msgs::CameraInfoConstPtr& l_camera_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_camera_info_msg)
  {
    vision_msgs::DetectionArray detections;
    detect(l_image_msg, r_image_msg, 
        l_camera_info_msg, r_camera_info_msg, detections);
    detections.header = l_image_msg->header;
    detections_pub_.publish(detections);
  }
   
 private:
  ros::NodeHandle nh_;
  ros::Publisher detections_pub_;
};

