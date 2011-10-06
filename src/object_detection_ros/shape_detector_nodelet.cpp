
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
        use_image_ = false;
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
        double matching_score_threshold, min_scale, max_scale;
        if (nh.getParam("matching_score_threshold", matching_score_threshold)) shape_detector_->setMatchingScoreThreshold(matching_score_threshold);
        if (nh.getParam("min_scale", min_scale)) shape_detector_->setMinScale(min_scale);
        if (nh.getParam("max_scale", max_scale)) shape_detector_->setMaxScale(max_scale);
      }

      void printSettings()
      {
        NODELET_INFO("Current ShapeDetector settings:\n"
                     "  Matching Score Threshold : %f \n"
                     "  Min Scale                : %f \n"
                     "  Max Scale                : %f \n",
                     shape_detector_->matchingScoreThreshold(),
                     shape_detector_->minScale(),
                     shape_detector_->maxScale());
      }
  };

}

#include <pluginlib/class_list_macros.h>
/**
* Pluginlib declaration. This is needed for the nodelet to be dynamically loaded/unloaded
*/
PLUGINLIB_DECLARE_CLASS (object_detection, ShapeDetector, object_detection_ros::ShapeDetectorNodelet, nodelet::Nodelet);

