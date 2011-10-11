#ifndef TRAININGDATA_H 
#define TRAININGDATA_H 

#include <vector>
#include <opencv2/core/core.hpp>

#include "odat/mask.h"
#include "odat/feature_set_3d.h"
#include "odat/pose2d.h"

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

    /// 2d pose of the object in the training image
    Pose2D image_pose;

    /// 3d feature set,
    /// the 3D coordinates are given in the camera frame
    FeatureSet3D features_3d;

    /// 3d pose of the object in the 3d features
    cv::Mat pose;
};

}


#endif /* TRAININGDATA_H */

