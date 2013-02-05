
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

#include "mono_detector.h"

class ColorShapeDetectorNode : public MonoDetector
{
private:
  boost::shared_ptr<object_detection::ColorDetector> color_detector_;
  boost::shared_ptr<object_detection::ShapeDetector> shape_detector_;

  ros::NodeHandle nh_priv_;

public:

  ColorShapeDetectorNode() :
    nh_priv_("~")
  {
    std::string db_type;
    nh_priv_.param<std::string>("db_type", db_type, "filesystem");
    std::string connection_string;
    if (!nh_priv_.getParam("connection_string", connection_string)) {
        ROS_ERROR("Parameter 'connection_string' is missing");
    }

    // instantiate the detector
    odat::ModelStorage::Ptr model_storage;
    if (db_type=="filesystem") {
        model_storage = boost::make_shared<odat::FilesystemModelStorage>(connection_string);
    }
    else {
      ROS_ERROR("Unknown model storage database type!");
    }
    // instantiate the detector
    color_detector_ = boost::make_shared<object_detection::ColorDetector>(model_storage);
    shape_detector_ = boost::make_shared<object_detection::ShapeDetector>(model_storage);

    std::string models;
    nh_priv_.getParam("models", models);
    if (models == "")
    {
      color_detector_->loadAllModels();
      shape_detector_->loadAllModels();
    }
    color_detector_->loadModelList(models);
    shape_detector_->loadModelList(models);

    loadSettings(nh_priv_);
    printSettings();
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

    // shape detector settings
    object_detection::ShapeDetector::Params shape_detector_params;
    nh.getParam("matching_score_threshold", shape_detector_params.matching_score_threshold);
    nh.getParam("min_scale", shape_detector_params.min_scale);
    nh.getParam("max_scale", shape_detector_params.max_scale);
    shape_detector_->setParams(shape_detector_params);
  }

  void printSettings()
  {
    ROS_INFO_STREAM("Current ColorDetector settings:\n" << color_detector_->params());
    std::vector<std::string> color_models = color_detector_->getLoadedModels();
    ROS_INFO_STREAM("Loaded " << color_models.size() << " color models.");
    for (size_t i = 0; i < color_models.size(); ++i) ROS_INFO_STREAM("    " << color_models[i]);
    
    ROS_INFO_STREAM("Current ShapeDetector settings:\n" << shape_detector_->params());
    std::vector<std::string> shape_models = shape_detector_->getLoadedModels();
    ROS_INFO_STREAM("Loaded " << shape_models.size() << " shape models.");
    for (size_t i = 0; i < shape_models.size(); ++i) ROS_INFO_STREAM("    " << shape_models[i]);
  }

  virtual void detect(
      const sensor_msgs::ImageConstPtr& image_msg, 
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
      vision_msgs::DetectionArray& detections_array)
  {
    detections_array.header = image_msg->header;
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    color_detector_->setImage(cv_ptr->image);
    color_detector_->detect();
    std::vector<odat::Detection> color_detections = color_detector_->getDetections();
    ROS_INFO_STREAM("ColorDetector made " << color_detections.size() << " detections.");
    if (color_detections.size() > 0)
    {
      shape_detector_->setInputDetections(color_detections);
      shape_detector_->detect();
      std::vector<odat::Detection> shape_detections = shape_detector_->getDetections();
      ROS_INFO_STREAM("ShapeDetector made " << shape_detections.size() << " detections.");
      if (shape_detections.size() > 0)
      {
        odat_ros::toMsg(shape_detections, detections_array);
      }
    }
  }

  
  virtual bool train(
      vision_msgs::TrainDetector::Request& training_request,
      vision_msgs::TrainDetector::Response& training_response)
  {
    training_response.success = false;
    ROS_INFO("Training service call received, running training.");
    if (training_request.outline.points.size() < 3)
    {
      training_response.message = "Not enough points in object outline polygon.";
      ROS_ERROR_STREAM(training_response.message);
      return false;
    }
    if (!sensor_msgs::image_encodings::isColor(training_request.image_left.encoding))
    {
      training_response.message = "Training image must be color!";
      ROS_ERROR_STREAM(training_response.message);
      return false;
    }
 
    odat::TrainingData training_data;
    boost::shared_ptr<void const> tracked_object;
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(training_request.image_left, tracked_object, sensor_msgs::image_encodings::BGR8);
    training_data.image = cv_ptr->image;
    training_data.mask.roi.x = 0;
    training_data.mask.roi.y = 0;
    training_data.mask.roi.width = training_data.image.cols;
    training_data.mask.roi.height = training_data.image.rows;
    training_data.mask.mask = cv::Mat(training_data.image.rows, training_data.image.cols, CV_8UC1, cv::Scalar::all(0));
    std::vector<cv::Point> points;
    for (size_t i = 0; i < training_request.outline.points.size(); ++i)
    {
      points.push_back(
          cv::Point(training_request.outline.points[i].x,
                    training_request.outline.points[i].y));
    }
    const cv::Point* point_data = points.data();
    int size = points.size();
    cv::fillPoly(training_data.mask.mask, &point_data, &size, 1, cv::Scalar::all(255));
    training_data.image_pose.x = training_request.image_pose.x;
    training_data.image_pose.y = training_request.image_pose.y;
    training_data.image_pose.theta = training_request.image_pose.theta;

    color_detector_->startTraining(training_request.object_id);
    color_detector_->trainInstance(training_request.object_id, training_data);
    color_detector_->endTraining(training_request.object_id);
    // run detection with color detector to get training data for shape detector
    color_detector_->setImage(training_data.image);
    color_detector_->detect();
    std::vector<odat::Detection> color_detections = color_detector_->getDetections();
    if (color_detections.size() == 0)
    {
      training_response.message = "Training error, color detector did not detect anything after training. Object cannot be trained.";
      ROS_ERROR_STREAM(training_response.message);
      return false;
    }
    // select detection of target
    int index = -1;
    for (size_t i = 0; i < color_detections.size(); ++i)
    {
      if (color_detections[i].object_id == training_request.object_id)
        index = i;
    }
    if (index < 0)
    {
      training_response.message = "Training error, color detector did not detect target after training. Object cannot be trained.";
      ROS_ERROR_STREAM(training_response.message);
      return false;
    }
    training_data.mask = color_detections[index].mask;
    shape_detector_->startTraining(training_request.object_id);
    shape_detector_->trainInstance(training_request.object_id, training_data);
    shape_detector_->endTraining(training_request.object_id);
    training_response.message = "Target trained.";
    training_response.success = true;
    ROS_INFO_STREAM(training_response.message);
    return true;
  }
  
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "object_detector");
  ColorShapeDetectorNode detector;
  ros::spin();
  return 0;
}

