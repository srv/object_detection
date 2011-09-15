#include <fstream>
#include <boost/algorithm/string.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>


#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/ros/register_point_struct.h>
#include <pcl/ros/conversions.h>
#include <pcl/point_types.h>

#include "object_detection/histogram_backprojection.h"
#include "object_detection/training_data.h"
#include "object_detection/detection.h"
#include "object_detection/utilities.h"
#include "object_detection/trainable.h"
#include "object_detection/detector.h"

#include "vision_msgs/TrainingData.h"
#include "vision_msgs/BoundingBoxStamped.h"
#include "vision_msgs/DetectionStamped.h"

namespace enc = sensor_msgs::image_encodings;
using object_detection::TrainingData;
using object_detection::Detector;
using object_detection::StereoFeature;


static const double DEFAULT_DETECTION_THRESHOLD = 0.5;

std::vector<cv::Point> readPolygonData(std::istream& in)
{
    std::vector<cv::Point> points;
    while (in.good())
    {
        int x, y;
        in >> x;
        in >> y;
        points.push_back(cv::Point(x, y));
    }
    return points;
}


class DetectorNode
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  
  // for synchronized subscriptions
  image_transport::SubscriberFilter sub_filter_image_;
  message_filters::Subscriber<vision_msgs::StereoFeatures> sub_filter_features_;
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
      vision_msgs::StereoFeatures> ExactPolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
  boost::shared_ptr<ExactSync> exact_sync_;
 
  ros::Subscriber training_data_sub_;
  ros::Publisher detection_pub_;

  bool is_trained_;

  Detector* detector_;

  double detection_threshold_;

public:
  DetectorNode()
    : nh_private_("~"), it_(nh_), is_trained_(false)
  {

    nh_private_.param<double>("detection_threshold", detection_threshold_, DEFAULT_DETECTION_THRESHOLD);

    ROS_INFO("detection_threshold set to %f", detection_threshold_);

    std::string detector_config_file;
    nh_private_.getParam("detector_config_file", detector_config_file);
    detector_ = new Detector(detector_config_file);

    // shall we use a training image from disk?
    bool use_training_image;
    nh_private_.param<bool>("use_training_image", use_training_image, false);
    if (use_training_image)
    {
        // load training parameters
        std::string image_file_name;
        std::string polygon_file_name;
        nh_private_.param<std::string>("image_file_name", image_file_name, "");
        nh_private_.param<std::string>("polygon_file_name", polygon_file_name, "");

        ROS_INFO("Using training image %s", image_file_name.c_str());
        ROS_INFO("Using polygon file %s", polygon_file_name.c_str());

        std::ifstream in(polygon_file_name.c_str());
        std::vector<cv::Point> object_polygon = readPolygonData(in);

        // perform training
        cv::Mat training_image = cv::imread(image_file_name);

        object_detection::TrainingData training_data;
        training_data.image = training_image;
        training_data.object_outline = object_polygon;

        train(training_data);

    } else
    {
        // subscribe to training data message
        training_data_sub_ = nh_.subscribe("training_data", 1, &DetectorNode::trainingDataCb, this);
        ROS_INFO("Waiting for training data message.");
    }
    image_sub_ = it_.subscribe("image", 1, &DetectorNode::imageCb, this);
    detection_pub_ = nh_private_.advertise<vision_msgs::DetectionStamped>("detection", 1);

    // Synchronize inputs. 
    exact_sync_.reset(new ExactSync(ExactPolicy(10),
                                    sub_filter_image_, sub_filter_features_));
    exact_sync_->registerCallback(
            boost::bind(&DetectorNode::detectionInputCb,
                        this, _1, _2));
    sub_filter_image_.subscribe(it_, "image", 1);
    sub_filter_features_.subscribe(nh_, "stereo_features", 1);

 }

  ~DetectorNode()
  {
      delete detector_;
  }

  void imageCb(const sensor_msgs::ImageConstPtr& image_msg)
  {
    cv::Mat image;
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(image_msg, enc::BGR8);
        image = cv_ptr->image.clone();
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    if (is_trained_)
    {
        std::vector<StereoFeature> stereo_features;
        std::vector<object_detection::Detection> detections = detector_->detect(image, stereo_features);
        cv::Mat image_with_detections = image.clone();

        // if we have one good detection
        if (detections.size() == 1 && detections[0].score > detection_threshold_)
        {
            publishDetection(detections[0], image_msg->header);
        }
    }
    cvWaitKey(3);
  }

  void detectionInputCb(const sensor_msgs::ImageConstPtr& image_msg,
          const vision_msgs::StereoFeaturesConstPtr& stereo_features_msg)
  {
    try
    {
        cv::Mat image;
        cv_bridge::CvImageConstPtr cv_ptr;
            cv_ptr = cv_bridge::toCvShare(image_msg, enc::BGR8);
            image = cv_ptr->image.clone();
        
        if (is_trained_)
        {
            std::vector<StereoFeature> stereo_features = 
                getStereoFeatures(*stereo_features_msg);
            std::vector<object_detection::Detection> detections = 
                detector_->detect(image, stereo_features);

            // if we have one good detection
            if (detections.size() == 1 && detections[0].score > detection_threshold_)
            {
                publishDetection(detections[0], image_msg->header);
            }
        }
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

    std::vector<StereoFeature> getStereoFeatures(const vision_msgs::StereoFeatures& stereo_features_msg)
    {
        if (stereo_features_msg.features.size() == 0)
        {
          return std::vector<StereoFeature>();
        }

        pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
        pcl::fromROSMsg(stereo_features_msg.world_points, point_cloud);
        assert(stereo_features_msg.features.size() == point_cloud.size());
        std::vector<StereoFeature> stereo_features(stereo_features_msg.features.size());
        pcl::PointCloud<pcl::PointXYZRGB>::const_iterator point_iter = 
            point_cloud.begin();
        for (size_t i = 0; i < stereo_features_msg.features.size(); ++i)
        {
            stereo_features[i].key_point.pt.x = stereo_features_msg.features[i].x;
            stereo_features[i].key_point.pt.y = stereo_features_msg.features[i].y;
            stereo_features[i].descriptor = stereo_features_msg.features[i].descriptor;
            const pcl::PointXYZRGB point = *point_iter;
            stereo_features[i].world_point.x = point.x;
            stereo_features[i].world_point.y = point.y;
            stereo_features[i].world_point.z = point.z;
            point_iter++;
        }
        return stereo_features;
    }

    void trainingDataCb(const vision_msgs::TrainingDataConstPtr& training_data_msg)
    {
        ROS_INFO("Training Data received");

        cv::Mat training_image;
        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(training_data_msg->image, training_data_msg, enc::BGR8);
            training_image = cv_ptr->image.clone();
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        vision_msgs::Polygon2D polygon = training_data_msg->object_description.outline;
        std::vector<cv::Point> object_polygon;
        for (size_t i = 0; i < polygon.points.size(); ++i)
        {
            object_polygon.push_back(cv::Point(polygon.points[i].x, polygon.points[i].y));
        }

        object_detection::TrainingData training_data;
        training_data.image = training_image;
        training_data.object_outline = object_polygon;

        // 3d information
        training_data.stereo_features =  
            getStereoFeatures(training_data_msg->stereo_features);

        train(training_data);

    }


    void train(const TrainingData& training_data)
    {
        if (training_data.isValid())
        {
            detector_->train(training_data);
            is_trained_ = true;
            ROS_INFO("Detector trained.");
        }
    }

    void publishDetection(const object_detection::Detection& detection, 
            const std_msgs::Header& header)
    {
        vision_msgs::Detection detection_msg;
        detection_msg.pose2D.x = detection.center.x;
        detection_msg.pose2D.y = detection.center.y;
        detection_msg.pose2D.theta = detection.angle;
        detection_msg.scale = detection.scale;
        detection_msg.id = detection.label;
        detection_msg.score = detection.score;

        vision_msgs::Polygon2D outline;
        for (size_t i = 0; i < detection.outline.size(); ++i)
        {
            vision_msgs::Point2D point;
            point.x = detection.outline[i].x;
            point.y = detection.outline[i].y;
            outline.points.push_back(point);
        }
        detection_msg.outline = outline;

        vision_msgs::DetectionStamped detection_msg_stamped;
        detection_msg_stamped.detection = detection_msg;
        detection_msg_stamped.header = header;
        detection_pub_.publish(detection_msg_stamped);
    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "object_detector");
  DetectorNode detector;
  ros::spin();
  return 0;
}

