#include <vision_msgs/DetectionArray.h>

#include "training_service.h"
#include "image_processor.h"

class MonoDetector :
  public TrainingService,
  public ImageProcessor
{
 public:
   MonoDetector() : TrainingService(), ImageProcessor()
   {
     detections_pub_ = nh_.advertise<vision_msgs::DetectionArray>("detections", 1);
   }

   virtual void detect(
      const sensor_msgs::ImageConstPtr& image_msg, 
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
      vision_msgs::DetectionArray& detections) = 0;

   virtual void imageCallback(
      const sensor_msgs::ImageConstPtr& image_msg, 
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg)
   {
     vision_msgs::DetectionArray detections;
     detect(image_msg, camera_info_msg, detections);
     detections_pub_.publish(detections);
   }
   
 private:
  ros::NodeHandle nh_;
  ros::Publisher detections_pub_;
};

