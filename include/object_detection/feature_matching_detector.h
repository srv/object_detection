#ifndef FEATURE_MATCHING_DETECTOR_H
#define FEATURE_MATCHING_DETECTOR_H

#include "odat/feature_set_3d.h"

namespace object_detection
{
  class FeatureMatchingDetector
  {
    public:

      struct Params
      {
        Params();
        std::string descriptor_matcher;
        double distance_ratio_threshold;
      };

      FeatureMatchingDetector();

      void setModelFeatures(const odat::FeatureSet3D& model)
      {
        model_features_ = model;
      }

      // set the k matrix
      void setCameraMatrix(const cv::Mat& k)
      {
        camera_matrix_k_ = k;
      }

      void detect(const odat::FeatureSet& features);

    private:

      /// the object model
      odat::FeatureSet3D model_features_;

      cv::Mat camera_matrix_k_;

      Params params_;
  };

} // end of namespace


#endif

