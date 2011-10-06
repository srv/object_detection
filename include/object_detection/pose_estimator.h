#ifndef POSE_ESTIMATOR_H
#define POSE_ESTIMATOR_H

#include <vector>

#include "trainable.h"

namespace object_detection {

struct Detection;
struct Pose;
class StereoFeature;

/**
 * \class PoseEstimator
 * \author Stephan Wirth
 * \brief Interface for pose estimators.
 * A pose estimator gets a detection and some input data and estimates
 * the pose of the detected object.
 */
class PoseEstimator : public Trainable
{
public:

  /**
    * Destructor
    */
  virtual ~PoseEstimator() {};

  /**
    * \brief Run the pose estimator.
    * \param detection input detection
    * \param image input image
    * \param stereo_features extracted stereo features
    * \return a pose for the detection
    */
  virtual Pose detect(const Detection& detection,
      const cv::Mat& image, 
      const std::vector<StereoFeature>& stereo_features) = 0;

  virtual void train(const TrainingData& training_data) = 0;

private:

};

}


#endif /* POSE_ESTIMATOR_H */

