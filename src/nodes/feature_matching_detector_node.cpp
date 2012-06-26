
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>

#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/ros/register_point_struct.h>
#include <pcl/ros/conversions.h>
#include <pcl/point_types.h>

#include <tf/transform_broadcaster.h>

#include <vision_msgs/Detection.h>

#include "object_detection/features_io.h"

namespace enc = sensor_msgs::image_encodings;

class FeatureMatchingDetectorNode
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber camera_sub_;

  cv::Mat model_descriptors_;
  std::vector<cv::Point3f> model_points_;

  cv::Ptr<cv::FeatureDetector> feature_detector_;
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor_;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;

  double matching_threshold_;
  std::string model_filename_;
  double sqr_unify_feature_distance_;

  tf::TransformBroadcaster tf_broadcaster_;

  ros::Publisher pose_pub_;
  ros::Publisher detection_pub_;

public:
  FeatureMatchingDetectorNode()
    : nh_private_("~"), it_(nh_)
  {

    std::string feature_detector, descriptor_extractor, descriptor_matcher;
    nh_private_.param("feature_detector", feature_detector, std::string("SURF"));
    nh_private_.param("descriptor_extractor", descriptor_extractor, std::string("SURF"));
    nh_private_.param("descriptor_matcher", descriptor_matcher, std::string("BruteForce"));
    double unify_feature_distance;
    nh_private_.param("unify_feature_distance", unify_feature_distance, 0.02);
    sqr_unify_feature_distance_ = unify_feature_distance * unify_feature_distance;

    feature_detector_ = cv::FeatureDetector::create(feature_detector);
    descriptor_extractor_ = cv::DescriptorExtractor::create(descriptor_extractor);
    descriptor_matcher_ = cv::DescriptorMatcher::create(descriptor_matcher);

    nh_private_.param("matching_threshold", matching_threshold_, 0.8);
    nh_private_.param("model_filename", model_filename_, std::string("model.yaml"));

    ROS_INFO_STREAM("Settings:" << std::endl
        << "  model_filename      : " << model_filename_ << std::endl
        << "  feature_detector    : " << feature_detector << std::endl
        << "  descriptor_extractor: " << descriptor_extractor << std::endl
        << "  descriptor_matcher  : " << descriptor_matcher << std::endl
        << "    matching_threshold: " << matching_threshold_ << std::endl);

    loadModel();

    camera_sub_ = it_.subscribeCamera("image", 1, 
        &FeatureMatchingDetectorNode::imageCb, this);

    pose_pub_ = nh_private_.advertise<geometry_msgs::PoseStamped>("target_pose", 1);
    detection_pub_ = nh_private_.advertise<vision_msgs::Detection>("detection", 1);

    ROS_INFO("Listening to %s", nh_.resolveName("image").c_str());

    cv::namedWindow("Features", 0);
  }

  ~FeatureMatchingDetectorNode()
  {
    cv::destroyWindow("Features");
  }

  void loadModel()
  {
    ROS_INFO("Loading model from '%s'", model_filename_.c_str());
    if (object_detection::features_io::loadFeatures(
          model_filename_, model_points_, model_descriptors_))
    {
      ROS_INFO("Loaded %zu points with %i descriptors.", model_points_.size(), model_descriptors_.rows);
    } else
    {
      ROS_ERROR("Could not load model!");
    }
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
    std::vector<cv::KeyPoint> keypoints;
    feature_detector_->detect(image, keypoints);
    cv::Mat descriptors;
    descriptor_extractor_->compute(image, keypoints, descriptors);

    // display
    cv::Mat canvas;
    cv::drawKeypoints(image, keypoints, canvas, 
        cv::Scalar(0, 255, 0), 4);
    cv::imshow("Features", canvas);
    cv::waitKey(3);


    double t2 = (double)cv::getTickCount();
    // match with model
    std::vector<std::vector<cv::DMatch> > knn_matches;
    descriptor_matcher_->knnMatch(descriptors, model_descriptors_, knn_matches, 2);

    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> world_points;
    for (size_t i = 0; i < knn_matches.size(); ++i)
    {
      if (knn_matches[i].size() == 2)
      {
         float score = knn_matches[i][0].distance / knn_matches[i][1].distance;
         cv::Point3f wp1 = model_points_[knn_matches[i][0].trainIdx];
         cv::Point3f wp2 = model_points_[knn_matches[i][1].trainIdx];
         float x_diff = wp1.x - wp2.x;
         float y_diff = wp1.y - wp2.x;
         float z_diff = wp1.z - wp2.x;
         float sqr_distance = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
         if (score < matching_threshold_ || sqr_distance < sqr_unify_feature_distance_)
         {
           image_points.push_back(keypoints[knn_matches[i][0].queryIdx].pt);
           world_points.push_back(model_points_[knn_matches[i][0].trainIdx]);
         }
      }
    }

    double t3 = (double)cv::getTickCount();
    if (world_points.size() > 5)
    {
      const cv::Mat P(3,4, CV_64FC1, const_cast<double*>(camera_info_msg->P.data()));
      const cv::Mat camera_matrix_k = P.colRange(cv::Range(0,3));
      cv::Mat distortion = cv::Mat::zeros(4, 1, CV_32F);
      cv::Mat r_vec(3, 1, CV_64FC1);
      cv::Mat t_vec(3, 1, CV_64FC1);
      bool use_extrinsic_guess = false;
      int num_iterations = 100;
      float allowed_reprojection_error = 8.0; // used by ransac to classify inliers
      int max_inliers = 100; // stop iteration if more inliers than this are found
      cv::Mat inliers;
      cv::solvePnPRansac(world_points, image_points, camera_matrix_k, 
          distortion, r_vec, t_vec, use_extrinsic_guess, num_iterations, 
          allowed_reprojection_error, max_inliers, inliers);
      int num_inliers = cv::countNonZero(inliers);
      int min_inliers = 8;
      if (num_inliers >= min_inliers)
      {
        std::cout << "Found transform with " << num_inliers 
          << " inliers from " << world_points.size() << " matches:"  << std::endl
          << "  r_vec: " << r_vec << std::endl
          << "  t_vec: " << t_vec << std::endl;

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

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "object_detector");
  FeatureMatchingDetectorNode detector;
  ros::spin();
  return 0;
}

