
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>
#include <vision_msgs/DetectionArray.h>

#include "odat/fs_model_storage.h"
#include "odat_ros/conversions.h"
#include "object_detection/color_detector.h"

namespace object_detection_ros {

  class ColorDetectorNodelet : public nodelet::Nodelet
  {
    private:
      boost::shared_ptr<object_detection::ColorDetector> color_detector_;

      image_transport::Subscriber image_sub_;
      boost::shared_ptr<image_transport::ImageTransport> it_;

      ros::Publisher detections_pub_;

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

        std::string models;
	    nh_priv.getParam("models", models);
	    color_detector_->loadModelList(models);

	    loadSettings(nh_priv);
	    printSettings();

        ros::NodeHandle& nh = getNodeHandle();
	    advertise(nh);
	    subscribe(nh);
      }  

      void loadSettings(ros::NodeHandle& nh)
      {
        // retrieve all parameters that are set (if getParam fails, the value is untouched)
        // color detector settings
        object_detection::ColorDetector::Params color_detector_params;
        nh.getParam("num_hue_bins", color_detector_params.num_hue_bins);
        nh.getParam("num_saturation_bins", color_detector_params.num_saturation_bins);
        nh.getParam("min_saturation", color_detector_params.min_saturation);
        nh.getParam("min_value", color_detector_params.min_value);
        nh.getParam("morph_element_size", color_detector_params.morph_element_size);
        nh.getParam("mean_filter_size", color_detector_params.mean_filter_size);
        nh.getParam("show_images", color_detector_params.show_images);
        color_detector_->setParams(color_detector_params);
      }

      void printSettings()
      {
        NODELET_INFO_STREAM("Current ColorDetector settings:\n" << color_detector_->params());
      }

      void advertise(ros::NodeHandle& nh)
      {
        detections_pub_ = nh.advertise<vision_msgs::DetectionArray>("detections", 1);
      }

      void subscribe(ros::NodeHandle& nh)
      {
        it_.reset(new image_transport::ImageTransport(nh));
        image_sub_ = it_->subscribe("image", 1, &ColorDetectorNodelet::detectionCb, this);
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
          std::vector<odat::Detection> detections = color_detector_->getDetections();
          vision_msgs::DetectionArrayPtr detections_msg(new vision_msgs::DetectionArray());
          odat_ros::toMsg(detections, *detections_msg);
          detections_msg->header = image_msg->header;
          detections_pub_.publish(detections_msg);
        }
      }
  };

}

#include <pluginlib/class_list_macros.h>
/**
* Pluginlib declaration. This is needed for the nodelet to be dynamically loaded/unloaded
*/
PLUGINLIB_DECLARE_CLASS (object_detection, ColorDetector, object_detection_ros::ColorDetectorNodelet, nodelet::Nodelet);

