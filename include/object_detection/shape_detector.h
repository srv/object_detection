#ifndef SHAPE_DETECTOR_H
#define SHAPE_DETECTOR_H

#include "odat/detector.h"
#include "odat/trainable.h"

namespace object_detection
{
  class ShapeDetector : public odat::Detector, public odat::Trainable
  {
    public:
      ShapeDetector(odat::ModelStorage::Ptr model_storage);

      virtual void detect();

      virtual std::string getName() { return "ShapeDetector"; }

      /**
      * Loads given models from model storage
      */
      virtual void loadModels(const std::vector<std::string>& models);

      /**
      * @return list of loaded models
      */
      virtual std::vector<std::string> getLoadedModels() const;

      /**
      * Saves the given model using model storage
      */
      void saveModel(const std::string& model);

      // re-implemented from Trainable
      virtual void startTraining(const std::string& name);
      virtual void trainInstance(const std::string& name, const odat::TrainingData& data);
      virtual void endTraining(const std::string& name);

      inline void setMatchingScoreThreshold(float threshold)
      {
        matching_score_threshold_ = threshold;
      }

      inline float matchingScoreThreshold() const { return matching_score_threshold_; }

      inline void setMinScale(float scale)
      {
        min_scale_ = scale;
      }

      inline float minScale() const { return min_scale_; }

      inline void setMaxScale(float scale)
      {
        max_scale_ = scale;
      }

      inline float maxScale() const { return max_scale_; }

    private:

      /// the object models
      std::map<std::string, std::vector<cv::Point> > model_shapes_;

      static const float DEFAULT_MATCHING_SCORE_THRESHOLD;
      static const float DEFAULT_MIN_SCALE;
      static const float DEFAULT_MAX_SCALE;

      /// detection limiting parameters
      float matching_score_threshold_;
      float min_scale_;
      float max_scale_;

  };

} // end of namespace


#endif

