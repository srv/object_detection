#ifndef TRAININGDATA_H 
#define TRAININGDATA_H 

#include <vector>
#include <opencv2/core/core.hpp>

#include "odat/mask.h"

namespace odat {

/**
 * \struct TrainingData
 * \author Stephan Wirth
 * \brief Data structure for training data that is used by Detectors.
 */
struct TrainingData 
{
    /// the image on which the object is visible
    cv::Mat image;

    /// region of interest that marks the object
    Mask mask; 
};

}


#endif /* TRAININGDATA_H */

