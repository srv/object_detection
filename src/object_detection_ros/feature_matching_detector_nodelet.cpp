
#include "odat_ros/detector_nodelet.h"
#include "odat/fs_model_storage.h"

#include "object_detection/feature_matching_detector.h"

namespace object_detection_ros {

  class FeatureMatchingDetectorNodelet : public odat_ros::DetectorNodelet
  {
    private:
      boost::shared_ptr<object_detection::FeatureMatchingDetector> feature_matching_detector_;

    public:
      /**
      * Initializes the nodelet
      */
      virtual void childInit(ros::NodeHandle& nh)
      {
        use_image_ = false;
        use_features_ = true;
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
        feature_matching_detector_ = boost::make_shared<object_detection::FeatureMatchingDetector>(model_storage);
        // save in variable that is used by base class
        detector_ = feature_matching_detector_;

        std::string models;
	    nh.getParam("models", models);
	    loadModels(models);

	    loadSettings(nh);
	    printSettings();
      }  

      void loadSettings(ros::NodeHandle& nh)
      {
      }

      void printSettings()
      {
      }
  };

}

#include <pluginlib/class_list_macros.h>
/**
* Pluginlib declaration. This is needed for the nodelet to be dynamically loaded/unloaded
*/
PLUGINLIB_DECLARE_CLASS (object_detection, FeatureMatchingDetector, object_detection_ros::FeatureMatchingDetectorNodelet, nodelet::Nodelet);

