
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "odat_ros/conversions.h"

void odat_ros::fromMsg(const vision_msgs::Detection& detection_msg, odat::Detection& detection)
{
  detection.label = detection_msg.label;
  detection.detector = detection_msg.detector;
  detection.score = detection_msg.score;
  fromMsg(detection_msg.mask, detection.mask);
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

void odat_ros::toMsg(const odat::Detection& detection, vision_msgs::Detection& detection_msg)
{
  detection_msg.label = detection.label;
  detection_msg.detector = detection.detector;
  detection_msg.score = detection.score;
  toMsg(detection.mask, detection_msg.mask);
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

