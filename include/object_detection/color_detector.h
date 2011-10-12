#ifndef COLOR_DETECTOR_H
#define COLOR_DETECTOR_H

#include "odat/detector.h"
#include "odat/trainable.h"

namespace object_detection
{
  class ColorDetector : public odat::Detector, public odat::Trainable
  {
    public:
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

      inline void setNumHueBins(int num_hue_bins)
      {
          assert(num_hue_bins > 0 && num_hue_bins <= 180);
          num_hue_bins_ = num_hue_bins;
      }

      inline int numHueBins() const { return num_hue_bins_; };

      inline void setNumSaturationBins(int num_saturation_bins)
      {
          assert(num_saturation_bins > 0 && num_saturation_bins <= 256);
          num_saturation_bins_ = num_saturation_bins;
      }

      inline int numSaturationBins() const { return num_saturation_bins_; }

      inline void setMinSaturation(int min_saturation)
      {
          assert(min_saturation >= 0 && min_saturation < 256);
          min_saturation_ = min_saturation;
      }

      inline int minSaturation() const { return min_saturation_; }

      inline void setMinValue(int min_value)
      {
          assert(min_value >= 0 && min_value < 256);
          min_value_ = min_value;
      }

      inline int minValue() const { return min_value_; }

      inline void setMorphElementSize(int element_size)
      {
          assert(element_size > 0 && element_size < 125);
          morph_element_size_ = element_size;
      }

      inline int morphElementSize() const { return morph_element_size_; }

      inline void setMeanFilterSize(int mean_filter_size)
      {
          assert(mean_filter_size > 0 && mean_filter_size % 2 == 1);
          mean_filter_size_ = mean_filter_size;
      }

      inline int meanFilterSize() const { return mean_filter_size_; }

      inline void setShowImages(bool show) { show_images_ = show; }

      inline bool showImages() const { return show_images_; }

    private:

      /// the object models
      std::map<std::string, cv::MatND> model_histograms_;

      /// training data temporal storage
      std::map<std::string, std::vector<odat::TrainingData> > training_data_;

      /// default value for number of hue bins
      static const int DEFAULT_NUM_HUE_BINS = 32;

      /// default value for number of saturation bins
      static const int DEFAULT_NUM_SATURATION_BINS = 32;

      /// default value for minimum saturation
      static const int DEFAULT_MIN_SATURATION = 50;

      /// default value for minimum value (brightness)
      static const int DEFAULT_MIN_VALUE = 20;

      /// default size of the element that is used in opening
      static const int DEFAULT_MORPH_ELEMENT_SIZE = 9;

      /// default size for mean filter
      static const int DEFAULT_MEAN_FILTER_SIZE = 1;

      /// number of bins for the hue channels,
      /// defaults to DEFAULT_NUM_HUE_BINS
      int num_hue_bins_;

      /// number of bins for the saturation channel, defaults to
      /// DEFAULT_NUM_SATURATION_BINS
      int num_saturation_bins_;

      /// minimum saturation that has to have a color to be
      /// part of the object model, defaults to DEFAULT_MIN_SATURATION
      int min_saturation_;

      /// minimum value that has to have a color to be part of the object
      /// model, defaults to DEFAULT_MIN_VALUE
      int min_value_;

      /// size of the element that is used in opening
      int morph_element_size_;

      /// size of the mean filter window
      int mean_filter_size_;

      /// show images while processing ?
      bool show_images_;

  };

} // end of namespace


#endif

