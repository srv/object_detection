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

    /// the rotation with respect to the trained object
    double angle;

    /// the object center 
    cv::Point center;

    /// the scale of the object with respect to the trained object
    double scale;
    
    /// some score (usually between 0 and 1) that tells the
    /// quality of the detection
    double score;
};

}


#endif /* DETECTION_H */
