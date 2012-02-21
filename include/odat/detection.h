#ifndef DETECTION_H
#define DETECTION_H

#include <iostream>
#include <string>

#include "odat/mask.h"
#include "odat/pose2d.h"

namespace odat {
  /**
  * \struct Detection
  * \author Stephan Wirth
  * \brief Represents a detection that was made by a Detector
  */
  struct Detection
  {
    /// mask that defines the location
    Mask mask;

    /// label to identify an object
    std::string label;

    /// names the detector that produced the detection
    std::string detector;

    /// some score (usually between 0 and 1) that tells the
    /// quality of the detection
    double score;

    /// describes the pose of the object in the image
    Pose2D image_pose;

    /// describes the pose of the object with respect to the sensor as
    /// 3D rigid transformation
    cv::Mat pose;

    /// describes the scaling of the object (in a 2D-based detection)
    double scale;

  };
}

std::ostream& operator<< (std::ostream& out, const odat::Detection& detection);

#endif /* DETECTION_H */

