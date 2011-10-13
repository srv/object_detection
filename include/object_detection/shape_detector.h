#ifndef SHAPE_DETECTOR_H
#define SHAPE_DETECTOR_H

#include "odat/detector.h"
#include "odat/trainable.h"

namespace object_detection
{
  /**
  * @brief Detector that compares shapes
  */
  class ShapeDetector : public odat::Detector, public odat::Trainable
  {
    public:

      /**
      * @brief Parameters for ShapeDetector
      */
      struct Params
      {
        /**
        * Creates a default parameter set
        */
        Params();

        /// threshold that defines a match
        double matching_score_threshold;
        
        /// minimum scaling of a shape that is accepted as detection
        double min_scale;
        
        /// maximum scaling of a shape that is accepted as detection
        double max_scale;

        /// defaults (defined in cpp)
        static const float DEFAULT_MATCHING_SCORE_THRESHOLD;
        static const float DEFAULT_MIN_SCALE;
        static const float DEFAULT_MAX_SCALE;
      };

      /**
      * Creates a shape detector with default parameters
      * @param model_storage storage to use to load/save models
      */
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

      /**
      * @param params new parameters
      */
      inline void setParams(const Params& params)
      {
        params_ = params;
      }

      /**
      * @return current parameters
      */
      inline Params params() const { return params_; }

    private:

      /// the object models
      std::map<std::string, std::vector<cv::Point> > model_shapes_;

      /// detection limiting parameters
      Params params_;

  };

} // end of namespace

std::ostream& operator<< (std::ostream& ostr, const object_detection::ShapeDetector::Params& params);

#endif

