
#include "odat_ros/detector_nodelet.h"
#include "odat/fs_model_storage.h"

#include "object_detection/shape_detector.h"

namespace object_detection_ros {

  class ShapeDetectorNodelet : public odat_ros::DetectorNodelet
  {
    private:
      boost::shared_ptr<object_detection::ShapeDetector> shape_detector_;

    public:
      /**
      * Initializes the nodelet
      */
      virtual void childInit(ros::NodeHandle& nh)
      {
        use_image_ = true;
        use_features_ = false;
        use_masks_ = false;
        use_input_detections_ = true;

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
        shape_detector_ = boost::make_shared<object_detection::ShapeDetector>(model_storage);
        // save in variable that is used by base class
        detector_ = shape_detector_;

        std::string models;
	    nh.getParam("models", models);
	    loadModels(models);

	    loadSettings(nh);
	    printSettings();
      }  

      void loadSettings(ros::NodeHandle& nh)
      {
        object_detection::ShapeDetector::Params params;
        nh.getParam("matching_score_threshold", params.matching_score_threshold);
        nh.getParam("min_scale", params.min_scale);
        nh.getParam("max_scale", params.max_scale);
        shape_detector_->setParams(params);
      }

      void printSettings()
      {
        NODELET_INFO_STREAM("Current ShapeDetector settings:\n" << shape_detector_->params());
      }
  };

}

#include <pluginlib/class_list_macros.h>
/**
* Pluginlib declaration. This is needed for the nodelet to be dynamically loaded/unloaded
*/
PLUGINLIB_DECLARE_CLASS (object_detection, ShapeDetector, object_detection_ros::ShapeDetectorNodelet, nodelet::Nodelet);

