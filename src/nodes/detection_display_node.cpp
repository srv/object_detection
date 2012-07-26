#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>
#include <image_geometry/pinhole_camera_model.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>

#include <vision_msgs/Detection.h>


class DetectionDisplayNode
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  image_transport::ImageTransport it_;

  ros::Subscriber detection_sub_;
  image_transport::CameraSubscriber camera_sub_;

  std::map<std::string, vision_msgs::DetectionConstPtr> detections_;

public:
  DetectionDisplayNode() :
    nh_private_("~"),
    it_(nh_)
  {
    detection_sub_ = 
      nh_.subscribe("detection", 1, &DetectionDisplayNode::detectionCallback, this);
    camera_sub_ = 
      it_.subscribeCamera("image", 1, &DetectionDisplayNode::imageCallback, this);
  }

  ~DetectionDisplayNode()
  {
  }

  void detectionCallback(const vision_msgs::DetectionConstPtr& detection_msg)
  {
    detections_[detection_msg->label] = detection_msg;
  }

  void imageCallback(const sensor_msgs::ImageConstPtr& image_msg,
                     const sensor_msgs::CameraInfoConstPtr& camera_info_msg)
  {
    cv::Mat image_with_detections;
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      namespace enc = sensor_msgs::image_encodings;
      cv_ptr = cv_bridge::toCvShare(image_msg, enc::BGR8);
      image_with_detections = cv_ptr->image.clone();
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    std::map<std::string, vision_msgs::DetectionConstPtr>::const_iterator iter;
    for (iter = detections_.begin(); iter != detections_.end(); ++iter)
    {
      if (iter->second->header.stamp + ros::Duration(5.0) > ros::Time::now())
      {
        paintDetection(image_with_detections, camera_info_msg, iter->second);
      }
    }
    cv::namedWindow("Detection Display");
    cv::imshow("Detection Display", image_with_detections);
    cv::waitKey(3);
  }

  void paintDetection(cv::Mat& image, 
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
      const vision_msgs::DetectionConstPtr& detection_msg)
  {
    cv::Point2d text_origin;
    if (detection_msg->pose.pose.position.z != 0)   // valid pose
    {
      image_geometry::PinholeCameraModel camera_model;
      camera_model.fromCameraInfo(camera_info_msg);
      cv::Point3d origin;
      origin.x = detection_msg->pose.pose.position.x;
      origin.y = detection_msg->pose.pose.position.y;
      origin.z = detection_msg->pose.pose.position.z;
      cv::Point2d origin2d = camera_model.project3dToPixel(origin);
      static const int RADIUS = 3;
      cv::circle(image, origin2d, RADIUS, CV_RGB(255,0,0), -1);
      text_origin = origin2d;
    }
    if (detection_msg->image_pose.x != 0 || detection_msg->image_pose.y != 0) // valid 2D pose
    {
      cv::Point center(detection_msg->image_pose.x, detection_msg->image_pose.y);
      cv::RotatedRect rect(center,
        cv::Size(50 * detection_msg->scale, 50 * detection_msg->scale),
        detection_msg->image_pose.theta / M_PI * 180.0);
      cv::ellipse(image, rect, cv::Scalar(0, 255, 0), 2);

      double radius = 20;
      cv::Point direction_point(radius * cos(detection_msg->image_pose.theta),
                                radius * sin(detection_msg->image_pose.theta));
      cv::line(image, center, center + direction_point, cv::Scalar(0, 255, 0), 3);
      cv::line(image, center, center + direction_point, cv::Scalar(0, 0, 255), 2);
      text_origin = center;
    }
    int baseline;
    static const int FONT = CV_FONT_HERSHEY_PLAIN;
    cv::Size text_size = 
      cv::getTextSize(detection_msg->label, FONT, 1.0, 1.0, &baseline);
    cv::Point origin(text_origin.x - text_size.width / 2,
                     text_origin.y - 3 - baseline - 3);
    cv::putText(image, detection_msg->label, origin, FONT, 1.0, CV_RGB(255,0,0));
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "detection_display");
  DetectionDisplayNode display;
  ros::spin();
  return 0;
}

