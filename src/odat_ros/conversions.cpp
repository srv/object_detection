
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "odat_ros/conversions.h"

void odat_ros::fromMsg(const vision_msgs::Detection& detection_msg, odat::Detection& detection)
{
  detection.label = detection_msg.label;
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

void odat_ros::fromMsg(const vision_msgs::Features& features_msg, odat::FeatureSet& features)
{
  features.key_points.resize(features_msg.key_points.size());
  for (size_t i = 0; i < features.key_points.size(); ++i)
  {
    features.key_points[i].pt.x = features_msg.key_points[i].x;
    features.key_points[i].pt.y = features_msg.key_points[i].y;
    features.key_points[i].size = features_msg.key_points[i].size;
    features.key_points[i].angle = features_msg.key_points[i].angle;
    features.key_points[i].response = features_msg.key_points[i].response;
    features.key_points[i].octave = features_msg.key_points[i].octave;
    const cv::Mat descriptor_mat(features_msg.descriptor_data, true);
    features.descriptors = descriptor_mat.reshape(0, features.key_points.size());
    features.descriptor_name = features_msg.descriptor_name;
  }
}

void odat_ros::fromMsg(const vision_msgs::Features3D& features_3d_msg, odat::FeatureSet3D& features_3d)
{
  fromMsg(features_3d_msg.features_left, features_3d.features_left);
  features_3d.world_points.resize(features_3d_msg.world_points.size());
  for (size_t i = 0; i < features_3d_msg.world_points.size(); ++i)
  {
    features_3d.world_points[i].x = features_3d_msg.world_points[i].x;
    features_3d.world_points[i].y = features_3d_msg.world_points[i].y;
    features_3d.world_points[i].z = features_3d_msg.world_points[i].z;
  }
}

void odat_ros::toMsg(const odat::Detection& detection, vision_msgs::Detection& detection_msg)
{
  detection_msg.label = detection.label;
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

