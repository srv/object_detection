#ifndef FEATURE_MATCHING_DETECTOR_H
#define FEATURE_MATCHING_DETECTOR_H

#include "odat/detector.h"
#include "odat/trainable.h"
#include "odat/feature_set_3d.h"

#include "object_detection/model3d.h"

namespace object_detection
{
  class FeatureMatchingDetector : public odat::Detector, public odat::Trainable
  {
    public:
      FeatureMatchingDetector(odat::ModelStorage::Ptr model_storage);

      virtual void detect();

      virtual std::string getName() { return "FeatureMatchingDetector"; }

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

    private:

      /// the object models
      // std::map<std::string, Model3D::Ptr> models_;

      /// training data temporal storage
      //std::map<std::string, std::vector<odat::TrainingData> > training_data_;

  };

} // end of namespace


#endif

