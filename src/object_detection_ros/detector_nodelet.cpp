
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>
#include <vision_msgs/DetectionArray.h>

#include "odat/fs_model_storage.h"
#include "odat_ros/conversions.h"
#include "object_detection/color_detector.h"
#include "object_detection/shape_detector.h"

namespace object_detection_ros {

  class DetectorNodelet : public nodelet::Nodelet
  {
    private:
      boost::shared_ptr<object_detection::ColorDetector> color_detector_;
      boost::shared_ptr<object_detection::ShapeDetector> shape_detector_;

      image_transport::Subscriber image_sub_;
      boost::shared_ptr<image_transport::ImageTransport> it_;

      ros::Publisher detections_pub_;

      ros::Subscriber training_data_sub_;

    public:
      virtual void onInit()
      {
        ros::NodeHandle& nh_priv = getPrivateNodeHandle();

        std::string db_type;
        nh_priv.param<std::string>("db_type", db_type, "filesystem");
        std::string connection_string;
        if (!nh_priv.getParam("connection_string", connection_string)) {
            NODELET_ERROR("Parameter 'connection_string' is missing");
        }

        // instantiate the detector
        odat::ModelStorage::Ptr model_storage;
        if (db_type=="filesystem") {
            model_storage = boost::make_shared<odat::FilesystemModelStorage>(connection_string);
        }
        else {
          NODELET_ERROR("Unknown model storage database type!");
        }
        // instantiate the detector
        color_detector_ = boost::make_shared<object_detection::ColorDetector>(model_storage);
        shape_detector_ = boost::make_shared<object_detection::ShapeDetector>(model_storage);

        std::string models;
	    nh_priv.getParam("models", models);
	    if (models == "")
        {
          color_detector_->loadAllModels();
          shape_detector_->loadAllModels();
        }
	    color_detector_->loadModelList(models);
	    shape_detector_->loadModelList(models);

	    loadSettings(nh_priv);
	    printSettings();

        ros::NodeHandle& nh = getNodeHandle();
	    advertise(nh_priv);
	    subscribe(nh);
      }  

      void loadSettings(ros::NodeHandle& nh)
      {
        // color detector settings
        int num_hue_bins, num_saturation_bins, min_saturation, min_value, morph_element_size, mean_filter_size;
        bool show_images;
        if (!nh.hasParam("num_hue_bins")) NODELET_ERROR("Param num_hue_bins not found!");
        if (nh.getParam("num_hue_bins", num_hue_bins)) color_detector_->setNumHueBins(num_hue_bins);
        if (nh.getParam("num_saturation_bins", num_saturation_bins)) color_detector_->setNumSaturationBins(num_saturation_bins);
        if (nh.getParam("min_saturation", min_saturation)) color_detector_->setMinSaturation(min_saturation);
        if (nh.getParam("min_value", min_value)) color_detector_->setMinValue(min_value);
        if (nh.getParam("morph_element_size", morph_element_size)) color_detector_->setMorphElementSize(morph_element_size);
        if (nh.getParam("mean_filter_size", mean_filter_size)) color_detector_->setMeanFilterSize(mean_filter_size);
        if (nh.getParam("show_images", show_images)) color_detector_->setShowImages(show_images);

        // shape detector settings
        double matching_score_threshold, min_scale, max_scale;
        if (nh.getParam("matching_score_threshold", matching_score_threshold)) shape_detector_->setMatchingScoreThreshold(matching_score_threshold);
        if (nh.getParam("min_scale", min_scale)) shape_detector_->setMinScale(min_scale);
        if (nh.getParam("max_scale", max_scale)) shape_detector_->setMaxScale(max_scale);
      }

      void printSettings()
      {
        NODELET_INFO("Current ColorDetector settings:\n"
                     "  Number of Hue Bins       : %i \n"
                     "  Number of Saturation Bins: %i \n"
                     "  Minimum Saturation       : %i \n"
                     "  Minimum Value            : %i \n"
                     "  Morph Element Size       : %i \n"
                     "  Mean Filter Size         : %i \n"
                     "  Show Images              : %s \n",
                     color_detector_->numHueBins(),
                     color_detector_->numSaturationBins(),
                     color_detector_->minSaturation(),
                     color_detector_->minValue(),
                     color_detector_->morphElementSize(),
                     color_detector_->meanFilterSize(),
                     (color_detector_->showImages() ? "true" : "false"));
        NODELET_INFO("Current ShapeDetector settings:\n"
                     "  Matching Score Threshold : %f \n"
                     "  Min Scale                : %f \n"
                     "  Max Scale                : %f \n",
                     shape_detector_->matchingScoreThreshold(),
                     shape_detector_->minScale(),
                     shape_detector_->maxScale());
        std::vector<std::string> color_models = color_detector_->getLoadedModels();
        NODELET_INFO_STREAM("Loaded " << color_models.size() << " color models.");
        for (size_t i = 0; i < color_models.size(); ++i) NODELET_INFO_STREAM("    " << color_models[i]);
        std::vector<std::string> shape_models = shape_detector_->getLoadedModels();
        NODELET_INFO_STREAM("Loaded " << shape_models.size() << " shape models.");
        for (size_t i = 0; i < shape_models.size(); ++i) NODELET_INFO_STREAM("    " << shape_models[i]);
      }

      void advertise(ros::NodeHandle& nh)
      {
        detections_pub_ = nh.advertise<vision_msgs::DetectionArray>("detections", 1);
      }

      void subscribe(ros::NodeHandle& nh)
      {
        it_.reset(new image_transport::ImageTransport(nh));
        image_sub_ = it_->subscribe("image", 1, &DetectorNodelet::detectionCb, this);

        training_data_sub_ = nh.subscribe("training_data", 1, &DetectorNodelet::trainingDataCb, this);
      }

      void detectionCb(const sensor_msgs::ImageConstPtr& image_msg)
      {
        if (detections_pub_.getNumSubscribers() > 0)
        {
          cv_bridge::CvImageConstPtr cv_ptr;
          try
          {
            cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
          }
          catch (cv_bridge::Exception& e)
          {
            NODELET_ERROR("cv_bridge exception: %s", e.what());
            return;
          }
          color_detector_->setImage(cv_ptr->image);
          color_detector_->detect();
          std::vector<odat::Detection> color_detections = color_detector_->getDetections();
          NODELET_INFO_STREAM("ColorDetector made " << color_detections.size() << " detections.");
          if (color_detections.size() > 0)
          {
            shape_detector_->setInputDetections(color_detections);
            shape_detector_->detect();
            std::vector<odat::Detection> shape_detections = shape_detector_->getDetections();
            NODELET_INFO_STREAM("ShapeDetector made " << shape_detections.size() << " detections.");
            if (shape_detections.size() > 0)
            {
              vision_msgs::DetectionArrayPtr detections_msg(new vision_msgs::DetectionArray());
              odat_ros::toMsg(shape_detections, *detections_msg);
              detections_msg->header = image_msg->header;
              detections_pub_.publish(detections_msg);
            }
          }
        }
      }

      
      void trainingDataCb(const vision_msgs::TrainingDataConstPtr& training_data_msg)
      {
        NODELET_INFO("Training data received, running training.");
        odat::TrainingData training_data;
        odat_ros::fromMsg(*training_data_msg, training_data);
        color_detector_->startTraining("target");
        color_detector_->trainInstance("target", training_data);
        color_detector_->endTraining("target");
        // run detection with color detector to get training data for shape detector
        color_detector_->setImage(training_data.image);
        color_detector_->detect();
        std::vector<odat::Detection> color_detections = color_detector_->getDetections();
        if (color_detections.size() == 0)
        {
          NODELET_ERROR("Training error, color detector did not detect anything after training. Object cannot be trained.");
          return;
        }
        // select detection of target
        int index = -1;
        for (size_t i = 0; i < color_detections.size(); ++i)
        {
          if (color_detections[i].label == "target")
            index = i;
        }
        if (index < 0)
        {
          NODELET_ERROR("Training error, color detector did not detect the target after training. Object cannot be trained.");
          return;
        }
        training_data.mask = color_detections[index].mask;
        shape_detector_->startTraining("target");
        shape_detector_->trainInstance("target", training_data);
        shape_detector_->endTraining("target");
        NODELET_INFO("Target trained.");
      }
     
  };

}

#include <pluginlib/class_list_macros.h>
/**
* Pluginlib declaration. This is needed for the nodelet to be dynamically loaded/unloaded
*/
PLUGINLIB_DECLARE_CLASS (object_detection, Detector, object_detection_ros::DetectorNodelet, nodelet::Nodelet);

