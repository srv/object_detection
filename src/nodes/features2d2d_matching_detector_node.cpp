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

#include "mono_detector.h"

namespace enc = sensor_msgs::image_encodings;

/**
 * Node that matches 2D features of the current image to 2D features
 * of a model. Works only on mostly planar objects/environments.
 * If a detection was made, the output is a homography.
 */
class Features2D2DMatchingDetectorNode : public MonoDetector
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  struct Model2D
  {
    cv::Mat image;
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> key_points;
    std::vector<cv::Point2f> outline;
    cv::Point2f origin;
    double theta;
  };

  feature_extraction::KeyPointDetector::Ptr key_point_detector_;
  feature_extraction::DescriptorExtractor::Ptr descriptor_extractor_;

  double matching_threshold_;

  tf::TransformBroadcaster tf_broadcaster_;

  Model2D model_;

public:
  Features2D2DMatchingDetectorNode()
    : MonoDetector(), nh_private_("~")
  {

    std::string key_point_detector, descriptor_extractor;
    nh_private_.param("key_point_detector", key_point_detector, std::string("CvORB"));
    nh_private_.param("descriptor_extractor", descriptor_extractor, std::string("CvORB"));

    key_point_detector_ = 
      feature_extraction::KeyPointDetectorFactory::create(key_point_detector);
    descriptor_extractor_ = 
      feature_extraction::DescriptorExtractorFactory::create(descriptor_extractor);

    nh_private_.param("matching_threshold", matching_threshold_, 0.8);
    std::string model_name;
    nh_private_.param("model", model_name, std::string(""));

    ROS_INFO_STREAM("Settings:" << std::endl
        << "  model               : " << model_name << std::endl
        << "  key_point_detector  : " << key_point_detector << std::endl
        << "  descriptor_extractor: " << descriptor_extractor << std::endl
        << "  matching_threshold  : " << matching_threshold_ << std::endl);

    if (model_name != "")
    {
      loadModel(model_name);
    }

    cv::namedWindow("Features", 0);
  }

  ~Features2D2DMatchingDetectorNode()
  {
    cv::destroyWindow("Features");
  }

  virtual bool train(
      vision_msgs::TrainDetector::Request& training_request,
      vision_msgs::TrainDetector::Response& training_response)
  {
    if (training_request.outline.points.size() < 3)
    {
      training_response.success = false;
      training_response.message = "Not enough points in object outline polygon.";
      return false;
    }
    try
    {
      boost::shared_ptr<void const> tracked_object;
      cv_bridge::CvImageConstPtr cv_ptr;
      cv_ptr = cv_bridge::toCvShare(training_request.image_left, tracked_object, enc::MONO8);
      cv::Mat image = cv_ptr->image;
      std::vector<cv::KeyPoint> key_points;
      cv::Mat descriptors;
      key_point_detector_->detect(image, key_points);

      // create mask to filter key points 
      std::vector<cv::Point> points(training_request.outline.points.size());
      std::vector<cv::Point2f> pointsf(training_request.outline.points.size());
      for (size_t i = 0; i < training_request.outline.points.size(); ++i)
      {
        points[i].x = static_cast<int>(training_request.outline.points[i].x);
        points[i].y = static_cast<int>(training_request.outline.points[i].y);
        pointsf[i].x = training_request.outline.points[i].x;
        pointsf[i].y = training_request.outline.points[i].y;
      }
      cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar::all(0));
      const cv::Point* point_data = points.data();
      int size = points.size();
      cv::Scalar color = cv::Scalar::all(255);
      cv::fillPoly(mask, &point_data, &size, 1, color);
      // filter key points
      cv::KeyPointsFilter::runByPixelsMask(key_points, mask);
      descriptor_extractor_->extract(image, key_points, descriptors);
      ROS_INFO("Training image has %zu features.", key_points.size());
      if (key_points.size() > 5)
      {
        model_.image = image;
        model_.key_points = key_points;
        model_.descriptors = descriptors;
        model_.outline = pointsf;
        model_.origin.x = training_request.image_pose.x;
        model_.origin.y = training_request.image_pose.y;
        model_.theta = training_request.image_pose.theta;
        training_response.success = true;
        training_response.message = "Model trained.";
        return true;
      }
      else
      {
        training_response.success = false;
        training_response.message = "Too few features in model.";
        return true;
      }
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      training_response.success = false;
      training_response.message = "Training data invalid, could not read left image.";
      return false;
    }
  }

  void loadModel(const std::string& model_name)
  {
    model_.image = cv::imread(model_name, 0);
    assert(!model_.image.empty());
    key_point_detector_->detect(model_.image, model_.key_points);
    descriptor_extractor_->extract(model_.image, model_.key_points, 
        model_.descriptors);
    ROS_INFO("Model has %zu features.", model_.key_points.size());
    model_.outline.push_back(cv::Point(0,0));
    model_.outline.push_back(cv::Point(0,model_.image.rows - 1));
    model_.outline.push_back(cv::Point(model_.image.cols - 1,model_.image.rows - 1));
    model_.outline.push_back(cv::Point(model_.image.cols - 1,0));
  }

  virtual void detect(
      const sensor_msgs::ImageConstPtr& image_msg, 
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
      vision_msgs::DetectionArray& detections_array)
  {
    if (model_.image.empty())
    {
      ROS_WARN("Detector has no model!");
      return;
    }
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
    std::vector<cv::KeyPoint> key_points;
    key_point_detector_->detect(image, key_points);
    cv::Mat descriptors;
    descriptor_extractor_->extract(image, key_points, descriptors);

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

    ROS_INFO("%4zu features, %4zu matches, %i inliers.", 
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

    if (!homography.empty())
    {
      std::vector<cv::Point2f> original_points = model_.outline;
      original_points.push_back(model_.origin);
      std::vector<cv::Point2f> transformed_points;
      cv::perspectiveTransform(original_points, transformed_points, homography);
      std::vector<cv::Point> paint_points(transformed_points.size() - 1);
      for (size_t i = 0; i < transformed_points.size() - 1; ++i)
      {
        paint_points[i].x = static_cast<int>(transformed_points[i].x);
        paint_points[i].y = static_cast<int>(transformed_points[i].y);
      }
      const cv::Point* point_data = &(paint_points[0]);
      int num_points = paint_points.size();
      cv::Scalar color(0, 255, 0);
      bool is_closed = true;
      cv::polylines(canvas, &point_data, &num_points, 1, is_closed, color);
      vision_msgs::Detection detection;
      detection.detector = "Features2D";
      detection.outline.points.resize(transformed_points.size());
      for (size_t i = 0; i < transformed_points.size(); ++i)
      {
        detection.outline.points[i].x = transformed_points[i].x;
        detection.outline.points[i].y = transformed_points[i].y;
        detection.outline.points[i].z = 0.0;
      }
      detection.homography.resize(homography.rows * homography.cols);
      for (int i = 0; i < homography.rows * homography.cols; ++i)
        detection.homography[i] = homography.at<double>(i);
      detection.image_pose.x = transformed_points[transformed_points.size() - 1].x;
      detection.image_pose.y = transformed_points[transformed_points.size() - 1].y;
      detection.image_pose.theta = 
        atan2(homography.at<double>(1, 0) - homography.at<double>(0, 1), 
              homography.at<double>(0, 0) + homography.at<double>(1, 1)) +
        model_.theta;
      if (detection.image_pose.theta > M_PI) detection.image_pose.theta -= 2*M_PI;
      if (detection.image_pose.theta < -M_PI) detection.image_pose.theta += 2*M_PI;
      detection.score = std::min(1.0, cv::countNonZero(inliers) * 0.01);
      detections_array.detections.push_back(detection);
      detections_array.header = image_msg->header;
    }

    imshow("Features", canvas);
    cv::waitKey(3);
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "features2d_object_detector");
  Features2D2DMatchingDetectorNode detector;
  ros::spin();
  return 0;
}

