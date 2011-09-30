#ifndef CONVERSIONS_H_
#define CONVERSIONS_H_

#include <vision_msgs/DetectionArray.h>
#include <vision_msgs/MaskArray.h>

#include "odat/detection.h"

namespace odat_ros
{
  void fromMsg(const vision_msgs::Detection& detection_msg, odat::Detection& detection);
  void fromMsg(const vision_msgs::DetectionArray& detections_msg, std::vector<odat::Detection>& detections);
  void fromMsg(const vision_msgs::Mask& mask_msg, odat::Mask& mask);
  void fromMsg(const vision_msgs::MaskArray& masks_msg, std::vector<odat::Mask>& masks);

  void toMsg(const odat::Detection& detection, vision_msgs::Detection& detection_msg);
  void toMsg(const std::vector<odat::Detection>& detections, vision_msgs::DetectionArray& detections_msg);
  void toMsg(const odat::Mask& mask, vision_msgs::Mask& mask_msg);
  void toMsg(const std::vector<odat::Mask>& masks, vision_msgs::MaskArray& masks_msg);
}

#endif

