#ifndef CONVERSIONS_H_
#define CONVERSIONS_H_

#include <vision_msgs/DetectionArray.h>
#include <vision_msgs/MaskArray.h>
#include <vision_msgs/Features.h>
#include <vision_msgs/Features3D.h>
#include <vision_msgs/TrainingData.h>

#include "odat/detection.h"
#include "odat/feature_set.h"
#include "odat/feature_set_3d.h"
#include "odat/training_data.h"

namespace odat_ros
{
  void fromMsg(const vision_msgs::Detection& detection_msg, odat::Detection& detection);
  void fromMsg(const vision_msgs::DetectionArray& detections_msg, std::vector<odat::Detection>& detections);
  void fromMsg(const vision_msgs::Mask& mask_msg, odat::Mask& mask);
  void fromMsg(const vision_msgs::MaskArray& masks_msg, std::vector<odat::Mask>& masks);
  void fromMsg(const vision_msgs::Features& features_msg, odat::FeatureSet& features);
  void fromMsg(const vision_msgs::Features3D& features3d_msg, odat::FeatureSet3D& features_3d);
  void fromMsg(const vision_msgs::TrainingData& training_data_msg, odat::TrainingData& training_data);

  void toMsg(const odat::Detection& detection, vision_msgs::Detection& detection_msg);
  void toMsg(const std::vector<odat::Detection>& detections, vision_msgs::DetectionArray& detections_msg);
  void toMsg(const odat::Mask& mask, vision_msgs::Mask& mask_msg);
  void toMsg(const std::vector<odat::Mask>& masks, vision_msgs::MaskArray& masks_msg);
  void toMsg(const odat::TrainingData& training_data, vision_msgs::TrainingData& training_data_msg);
}

#endif

