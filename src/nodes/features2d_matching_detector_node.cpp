
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>

#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <feature_extraction/key_point_detector_factory.h>
#include <feature_extraction/descriptor_extractor_factory.h>
#include <feature_matching/matching_methods.h>

#include <tf/transform_broadcaster.h>

#include <vision_msgs/Detection.h>

namespace enc = sensor_msgs::image_encodings;

/**
 * Node that matches 2D features of the current image to 2D features
 * of a model. Works only on mostly planar objects/environments.
 * If a detection was made, the output is x/y/theta/scale.
 */
class Features2DMatchingDetectorNode
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber camera_sub_;

  struct Model2D
  {
    cv::Mat image;
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> key_points;
    std::vector<cv::Point> outline;
    cv::Point origin;
    double theta;
  };

  feature_extraction::KeyPointDetector::Ptr key_point_detector_;
  feature_extraction::DescriptorExtractor::Ptr descriptor_extractor_;

  double matching_threshold_;

  tf::TransformBroadcaster tf_broadcaster_;

  ros::Publisher pose_pub_;
  ros::Publisher detection_pub_;

  Model2D model_;

public:
  Features2DMatchingDetectorNode()
    : nh_private_("~"), it_(nh_)
  {

    std::string key_point_detector, descriptor_extractor;
    nh_private_.param("key_point_detector", key_point_detector, std::string("SmartSURF"));
    nh_private_.param("descriptor_extractor", descriptor_extractor, std::string("SmartSURF"));

    key_point_detector_ = 
      feature_extraction::KeyPointDetectorFactory::create(key_point_detector);
    descriptor_extractor_ = 
      feature_extraction::DescriptorExtractorFactory::create(descriptor_extractor);

    nh_private_.param("matching_threshold", matching_threshold_, 0.8);
    std::string model_name;
    nh_private_.param("model", model_name, std::string("target"));

    ROS_INFO_STREAM("Settings:" << std::endl
        << "  model               : " << model_name << std::endl
        << "  key_point_detector  : " << key_point_detector << std::endl
        << "  descriptor_extractor: " << descriptor_extractor << std::endl
        << "  matching_threshold  : " << matching_threshold_ << std::endl);

    loadModel(model_name);

    camera_sub_ = it_.subscribeCamera("image", 1, 
        &Features2DMatchingDetectorNode::imageCb, this);

    pose_pub_ = nh_private_.advertise<geometry_msgs::PoseStamped>("target_pose", 1);
    detection_pub_ = nh_private_.advertise<vision_msgs::Detection>("detection", 1);

    ROS_INFO("Listening to %s", nh_.resolveName("image").c_str());

    cv::namedWindow("Features", 0);
  }

  ~Features2DMatchingDetectorNode()
  {
    cv::destroyWindow("Features");
  }

  void loadModel(const std::string& model_name)
  {
    model_.image = cv::imread(model_name, 0);
    assert(!model_.image.empty());
    key_point_detector_->detect(model_.image, model_.key_points);
    descriptor_extractor_->extract(model_.image, model_.key_points, 
        model_.descriptors);
    ROS_INFO("Model has %zu features.", model_.key_points.size());
    model_.origin.x = model_.image.cols / 2;
    model_.origin.y = model_.image.rows / 2;
    /*
    ROS_INFO("Loading model from '%s'", model_filename_.c_str());
    if (object_detection::features_io::loadFeatures(
          model_filename_, model_points_, model_descriptors_))
    {
      ROS_INFO("Loaded %zu points with %i descriptors.", model_points_.size(), model_descriptors_.rows);
    } else
    {
      ROS_ERROR("Could not load model!");
    }
    */
  }

  void imageCb(const sensor_msgs::ImageConstPtr& image_msg, 
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg)
  {
    cv::Mat image;
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(image_msg, enc::MONO8);
        image = cv_ptr->image;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // detect features on image
    double t1 = (double)cv::getTickCount();
    std::vector<cv::KeyPoint> key_points;
    key_point_detector_->detect(image, key_points);
    cv::Mat descriptors;
    descriptor_extractor_->extract(image, key_points, descriptors);


    double t2 = (double)cv::getTickCount();
    // match with model
    std::vector<cv::DMatch> matches;
    cv::Mat match_mask;
    feature_matching::matching_methods::crossCheckThresholdMatching(descriptors, 
        model_.descriptors, matching_threshold_, match_mask, matches);

    std::vector<int> query_indices(matches.size()), train_indices(matches.size());
    for (size_t i = 0; i < matches.size(); i++)
    {
      query_indices[i] = matches[i].queryIdx;
      train_indices[i] = matches[i].trainIdx;
    }

    double t3 = (double)cv::getTickCount();
    cv::Mat inliers;
    cv::Mat homography;
    if (matches.size() > 10)
    {
      std::vector<cv::Point2f> points1; 
      cv::KeyPoint::convert(key_points, points1, query_indices);
      std::vector<cv::Point2f> points2; 
      cv::KeyPoint::convert(model_.key_points, points2, train_indices);
      double reprojection_threshold = 2;
      homography = cv::findHomography(
          cv::Mat(points2), cv::Mat(points1), CV_RANSAC, reprojection_threshold, 
          inliers);
    }

    ROS_INFO("%4zu features, %4zu matches, %4zu inliers.", 
        key_points.size(), matches.size(), cv::countNonZero(inliers));


    cv::Mat canvas;
      
    // draw all matches
    cv::drawMatches(
        image, key_points, model_.image, model_.key_points, matches, canvas, 
        CV_RGB(255, 0, 0), CV_RGB(255, 0, 0), std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // draw inliers
    cv::drawMatches(
        image, key_points, model_.image, model_.key_points, matches, canvas, 
        CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), inliers,
        cv::DrawMatchesFlags::DRAW_OVER_OUTIMG /*| 
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS*/);

    if(!homography.empty())
    {
      cv::Point2f origin = model_.origin;
      double direction = model_.theta;
      cv::Point2f x_axis(50 * cos(direction), 50 * sin(direction));
      cv::Point2f y_axis(50 * cos(direction + M_PI_2), 50 * sin(direction + M_PI_2));
      std::vector<cv::Point2f> points(3);
      points[0] = origin;
      points[1] = origin + x_axis;
      points[2] = origin + y_axis;
      std::vector<cv::Point2f> transformed_points;
      cv::perspectiveTransform(points, transformed_points, homography);
      
      cv::line(canvas, transformed_points[0], transformed_points[1], cv::Scalar(0, 0, 255), 2);
      cv::line(canvas, transformed_points[0], transformed_points[2], cv::Scalar(0, 255, 0), 2);
    }

    imshow("Features", canvas);
    cv::waitKey(3);
}

/*

        // publish result
        ros::Time stamp = image_msg->header.stamp;
        if (stamp.toSec()==0.0)
          stamp = ros::Time::now();
        sendMessageAndTransform(t_vec, r_vec, stamp, image_msg->header.frame_id);
      }
      else
      {
        ROS_INFO("Not enough inliers (%i) in %zu matches. Minimum is %i.", 
            num_inliers, world_points.size(), min_inliers);
      }
    } 
    else
    {
      ROS_INFO("Not enough matches (%zu).", world_points.size());
    }

    double t4 = (double)cv::getTickCount();
    double detect_time   = (t2 - t1) / cv::getTickFrequency() * 1000;
    double match_time    = (t3 - t2) / cv::getTickFrequency() * 1000;
    double pose_est_time = (t4 - t3) / cv::getTickFrequency() * 1000;

    double total = (t4 - t1) / cv::getTickFrequency() * 1000;

    ROS_INFO_STREAM("Times (total: " << total << "ms): detection: " << detect_time
        << "ms, matching: " << match_time << ", pose estimation: " << pose_est_time);
  }

  void sendMessageAndTransform(const cv::Mat& t_vec, const cv::Mat& r_vec, 
      const ros::Time& stamp, const std::string& camera_frame_id)
  {
    tf::Vector3 axis(
        r_vec.at<double>(0, 0), r_vec.at<double>(1, 0), r_vec.at<double>(2, 0));
    double angle = cv::norm(r_vec);
    tf::Quaternion quaternion(axis, angle);
    
    tf::Vector3 translation(
        t_vec.at<double>(0, 0), t_vec.at<double>(1, 0), t_vec.at<double>(2, 0));

    tf::Transform transform(quaternion, translation);
    tf::StampedTransform stamped_transform(
        transform, stamp, camera_frame_id, "/target");
    tf_broadcaster_.sendTransform(stamped_transform);

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = camera_frame_id;
    tf::poseTFToMsg(transform, pose_msg.pose);

    pose_pub_.publish(pose_msg);

    vision_msgs::Detection detection_msg;
    detection_msg.header.stamp = stamp;
    detection_msg.header.frame_id = camera_frame_id;
    detection_msg.label = model_filename_;
    detection_msg.detector = "feature_matching_detector";
    detection_msg.pose.pose = pose_msg.pose;
    detection_pub_.publish(detection_msg);
  }
  */

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "object_detector");
  Features2DMatchingDetectorNode detector;
  ros::spin();
  return 0;
}

