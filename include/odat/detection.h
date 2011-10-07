#ifndef DETECTION_H
#define DETECTION_H

#include <iostream>
#include <string>

#include "odat/mask.h"

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

    /// describes the transformation of the detected
    /// object relative to the training data.
    /// this can be a 6D transformation (3D manifold) (3x4 matrix)
    /// or a 3D transformation (2D manifold) (2x3 matrix)
    cv::Mat transform;

    /// describes the scaling of the object (in a 2D-based detection
    double scale;

  };
}

std::ostream& operator<< (std::ostream& out, const odat::Detection& detection);

#endif /* DETECTION_H */

