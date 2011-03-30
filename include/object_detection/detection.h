#ifndef DETECTION_H
#define DETECTION_H

#include <string>
#include <cv.h>

namespace object_detection {

/**
 * \struct Detection
 * \author Stephan Wirth
 * \brief Represents a detection that was made by a Detector
 */
struct Detection
{
    /// label to identify an object
    std::string label;

    /// the rotated bounding rectangle
    cv::RotatedRect bounding_rotated_rect;
    
    /// some score (usually between 0 and 1) that tells the
    /// quality of the detection
    double score;
};

}


#endif /* DETECTION_H */
