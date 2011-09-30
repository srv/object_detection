
#include "odat_ros/detector_nodelet.h"
#include "odat/fs_model_storage.h"

#include "object_detection/color_detector.h"

namespace object_detection_ros {

  class ColorDetectorNodelet : public odat_ros::DetectorNodelet
  {
    private:
      boost::shared_ptr<object_detection::ColorDetector> color_detector_;

    public:
      /**
      * Initializes the nodelet
      */
      virtual void childInit(ros::NodeHandle& nh)
      {
        use_image_ = true;
        use_point_cloud_ = false;
        use_masks_ = false;
        use_input_detections_ = false;

        std::string db_type;
        nh.param<std::string>("db_type", db_type, "filesystem");
        std::string connection_string;
        if (!nh.getParam("connection_string", connection_string)) {
            NODELET_ERROR("Parameter 'connection_string' is missing");
        }

        // instantiate the detector
        odat::ModelStorage::Ptr model_storage;
        if (db_type=="filesystem") {
            model_storage = boost::make_shared<odat::FilesystemModelStorage>(connection_string);
        }
        else {
          NODELET_ERROR("Unknown model storage database type!");
          //model_storage = boost::make_shared<rein::DatabaseModelStorage>(db_type,connection_string);
        }
        // instantiate the detector
        color_detector_ = boost::make_shared<object_detection::ColorDetector>(model_storage);
        // save in variable that is used by base class
        detector_ = color_detector_;

        std::string models;
	    nh.getParam("models", models);
	    if (models == "")
        {
          NODELET_INFO("Loading all models.");
        }
        else
        {
          NODELET_INFO("Loading the following models: %s", models.c_str());
        }
	    loadModels(models);
	    NODELET_INFO("Loading done.");

	    loadSettings(nh);
	    printSettings();
      }  

      void loadSettings(ros::NodeHandle& nh)
      {
        int num_hue_bins, num_saturation_bins, min_saturation, min_value, morph_element_size;
        bool show_images;
        if (!nh.hasParam("num_hue_bins")) NODELET_ERROR("Param num_hue_bins not found!");
        if (nh.getParam("num_hue_bins", num_hue_bins)) color_detector_->setNumHueBins(num_hue_bins);
        if (nh.getParam("num_saturation_bins", num_saturation_bins)) color_detector_->setNumSaturationBins(num_saturation_bins);
        if (nh.getParam("min_saturation", min_saturation)) color_detector_->setMinSaturation(min_saturation);
        if (nh.getParam("min_value", min_value)) color_detector_->setMinValue(min_value);
        if (nh.getParam("morph_element_size", morph_element_size)) color_detector_->setMorphElementSize(morph_element_size);
        if (nh.getParam("show_images", show_images)) color_detector_->setShowImages(show_images);
      }

      void printSettings()
      {
        NODELET_INFO("Current ColorDetector settings:\n"
                     "  Number of Hue Bins       : %i \n"
                     "  Number of Saturation Bins: %i \n"
                     "  Minimum Saturation       : %i \n"
                     "  Minimum Value            : %i \n"
                     "  Morph Element Size       : %i \n"
                     "  Show Images              : %s \n",
                     color_detector_->numHueBins(),
                     color_detector_->numSaturationBins(),
                     color_detector_->minSaturation(),
                     color_detector_->minValue(),
                     color_detector_->morphElementSize(),
                     (color_detector_->showImages() ? "true" : "false"));
      }
  };

}

#include <pluginlib/class_list_macros.h>
/**
* Pluginlib declaration. This is needed for the nodelet to be dynamically loaded/unloaded
*/
PLUGINLIB_DECLARE_CLASS (object_detection, ColorDetector, object_detection_ros::ColorDetectorNodelet, nodelet::Nodelet);

