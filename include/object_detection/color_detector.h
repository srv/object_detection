#ifndef COLOR_DETECTOR_H
#define COLOR_DETECTOR_H

#include "odat/detector.h"
#include "odat/trainable.h"

namespace object_detection
{
  /**
  * @brief Detector that compares colors
  */
  class ColorDetector : public odat::Detector, public odat::Trainable
  {
    public:

      /**
      * @brief Parameters for the ColorDetector
      */
      struct Params
      {
        /**
        * Default constructor sets all values to defaults.
        */
        Params();

        /// number of bins for the hue channels
        int num_hue_bins;

        /// number of bins for the saturation channel
        int num_saturation_bins;

        /// minimum saturation that has to have a color to be part of the object model
        int min_saturation;

        /// minimum value that has to have a color to be part of the object model
        int min_value;

        /// size of the element that is used in opening
        int morph_element_size;

        /// size of the mean filter window, must be odd and >0, a size of 1 means no filter is used
        int mean_filter_size;

        /// show images while processing? default = false
        bool show_images;

        static const int DEFAULT_NUM_HUE_BINS = 32;
        static const int DEFAULT_NUM_SATURATION_BINS = 32;
        static const int DEFAULT_MIN_SATURATION = 50;
        static const int DEFAULT_MIN_VALUE = 20;
        static const int DEFAULT_MORPH_ELEMENT_SIZE = 9;
        static const int DEFAULT_MEAN_FILTER_SIZE = 1;
      };

      /**
      * Creates a detector with default parameters that uses given model storage
      * \param model_storage storage to use to save/load models
      */
      ColorDetector(odat::ModelStorage::Ptr model_storage);

      /**
      * Adapts given model histogram according to the current settings
      * (number of bins, min saturation)
      */
      cv::MatND adaptHistogram(const std::string& model_name, const cv::MatND& model_histogram);

      virtual void detect();

      virtual std::string getName() { return "ColorDetector"; }

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
      std::map<std::string, cv::MatND> model_histograms_;

      /// training data temporal storage
      std::map<std::string, std::vector<odat::TrainingData> > training_data_;

      /// stores parameters
      Params params_;

  };

} // end of namespace


/// output operator for color detector parameters
std::ostream& operator<< (std::ostream& ostr, const object_detection::ColorDetector::Params& params);

#endif

