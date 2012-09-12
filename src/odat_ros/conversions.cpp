
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "odat_ros/conversions.h"

void odat_ros::fromMsg(const vision_msgs::Detection& detection_msg, odat::Detection& detection)
{
  detection.object_id = detection_msg.object_id;
  detection.detector = detection_msg.detector;
  detection.score = detection_msg.score;
  fromMsg(detection_msg.mask, detection.mask);
  detection.image_pose.x = detection_msg.image_pose.x;
  detection.image_pose.y = detection_msg.image_pose.y;
  detection.image_pose.theta = detection_msg.image_pose.theta;
  detection.scale = detection_msg.scale;
  // TODO 3D pose!!
}

void odat_ros::fromMsg(const vision_msgs::DetectionArray& detections_msg, std::vector<odat::Detection>& detections)
{
  detections.resize(detections_msg.detections.size());
  for (size_t i = 0; i < detections.size(); ++i)
  {
    fromMsg(detections_msg.detections[i], detections[i]);
  }
}

void odat_ros::fromMsg(const vision_msgs::Mask& mask_msg, odat::Mask& mask)
{
  mask.roi.x = mask_msg.roi.x;
  mask.roi.y = mask_msg.roi.y;
  mask.roi.width = mask_msg.roi.width;
  mask.roi.height = mask_msg.roi.height;
  cv_bridge::CvImageConstPtr cv_image = cv_bridge::toCvCopy(mask_msg.mask);
  mask.mask = cv_image->image;
}

void odat_ros::fromMsg(const vision_msgs::MaskArray& masks_msg, std::vector<odat::Mask>& masks)
{
  masks.resize(masks_msg.masks.size());
  for (size_t i = 0; i < masks.size(); ++i)
  {
    fromMsg(masks_msg.masks[i], masks[i]);
  }  
}

void odat_ros::fromMsg(const vision_msgs::TrainingData& training_data_msg, odat::TrainingData& training_data)
{
  assert(training_data_msg.image.encoding == "bgr8");
  cv_bridge::CvImageConstPtr cv_image = cv_bridge::toCvCopy(training_data_msg.image);
  training_data.image = cv_image->image;
  fromMsg(training_data_msg.mask, training_data.mask);
  training_data.image_pose.x = training_data_msg.image_pose.x;
  training_data.image_pose.y = training_data_msg.image_pose.y;
  training_data.image_pose.theta = training_data_msg.image_pose.theta;
}

void odat_ros::toMsg(const odat::Detection& detection, vision_msgs::Detection& detection_msg)
{
  detection_msg.object_id= detection.object_id;
  detection_msg.detector = detection.detector;
  detection_msg.score = detection.score;
  toMsg(detection.mask, detection_msg.mask);
  detection_msg.image_pose.x = detection.image_pose.x;
  detection_msg.image_pose.y = detection.image_pose.y;
  detection_msg.image_pose.theta = detection.image_pose.theta;
  detection_msg.scale = detection.scale;
  // TODO 3d pose
}

void odat_ros::toMsg(const std::vector<odat::Detection>& detections, vision_msgs::DetectionArray& detections_msg)
{
  detections_msg.detections.resize(detections.size());
  for (size_t i = 0; i < detections_msg.detections.size(); ++i)
  {
    toMsg(detections[i], detections_msg.detections[i]);
  }
}

void odat_ros::toMsg(const odat::Mask& mask, vision_msgs::Mask& mask_msg)
{
  mask_msg.roi.x = mask.roi.x;
  mask_msg.roi.y = mask.roi.y;
  mask_msg.roi.width = mask.roi.width;
  mask_msg.roi.height = mask.roi.height;

  cv_bridge::CvImage cv_image;
  cv_image.image = mask.mask;
  cv_image.encoding = sensor_msgs::image_encodings::MONO8;
  cv_image.toImageMsg(mask_msg.mask);
}

void odat_ros::toMsg(const std::vector<odat::Mask>& masks, vision_msgs::MaskArray& masks_msg)
{
  masks_msg.masks.resize(masks.size());
  for (size_t i = 0; i < masks_msg.masks.size(); ++i)
  {
    toMsg(masks[i], masks_msg.masks[i]);
  }
}

void odat_ros::toMsg(const odat::TrainingData& training_data, vision_msgs::TrainingData& training_data_msg)
{
  assert(training_data.image.type() == CV_8UC3);
  cv_bridge::CvImage cv_image;
  cv_image.image = training_data.image;
  cv_image.encoding = sensor_msgs::image_encodings::BGR8;
  cv_image.toImageMsg(training_data_msg.image);
  toMsg(training_data.mask, training_data_msg.mask);
  training_data_msg.image_pose.x = training_data.image_pose.x;
  training_data_msg.image_pose.y = training_data.image_pose.y;
  training_data_msg.image_pose.theta = training_data.image_pose.theta;
  // TODO 3D info
}

