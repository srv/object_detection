#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <feature_extraction/key_point_detector_factory.h>
#include <feature_extraction/descriptor_extractor_factory.h>
#include <feature_matching/matching_methods.h>
#include <feature_matching/stereo_feature_matcher.h>
#include <feature_matching/stereo_depth_estimator.h>

#include <tf/transform_datatypes.h>

#include "stereo_detector.h"

namespace enc = sensor_msgs::image_encodings;

/**
 * Node that matches 3D features of the current image pair to 2D features
 * of a model. If the object/environment is mostly planar, a homography
 * will be computed. In all cases the pose of the camera taking the
 * training image will be estimated and published.
 */
class Features2D3DDetectorNode : public StereoDetector
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Publisher features_image_pub_;

  struct Model2D
  {
    cv::Mat image;
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> key_points;
    std::vector<cv::Point2f> outline;
    cv::Point2f origin;
    double theta;
    std::string object_id;
  };

  feature_extraction::KeyPointDetector::Ptr key_point_detector_;
  feature_extraction::DescriptorExtractor::Ptr descriptor_extractor_;

  double matching_threshold_;
  double stereo_matching_threshold_;
  int min_model_features_count_;
  bool equalize_histogram_;
  bool normalize_illumination_;
  int min_num_matches_;

  Model2D model_;

  cv::Mat illumination_;
  cv::Scalar mean_illumination_;

public:
  Features2D3DDetectorNode()
    : StereoDetector(), nh_private_("~")
  {

    std::string key_point_detector, descriptor_extractor;
    nh_private_.param("key_point_detector", key_point_detector, std::string("CvORB"));
    nh_private_.param("descriptor_extractor", descriptor_extractor, std::string("CvORB"));
    nh_private_.param("min_model_features_count", min_model_features_count_, 20);
    nh_private_.param("equalize_histogram", equalize_histogram_, false);
    nh_private_.param("normalize_illumination", normalize_illumination_, false);
    nh_private_.param("min_num_matches", min_num_matches_, 7);
    if (min_num_matches_ < 4)
    {
      ROS_WARN("min_num_matches must be > 4, setting to 4!");
      min_num_matches_ = 4;
    }

    features_image_pub_ = nh_private_.advertise<sensor_msgs::Image>("features", 1);

    key_point_detector_ = 
      feature_extraction::KeyPointDetectorFactory::create(key_point_detector);
    descriptor_extractor_ = 
      feature_extraction::DescriptorExtractorFactory::create(descriptor_extractor);

    nh_private_.param("matching_threshold", matching_threshold_, 0.8);
    nh_private_.param("stereo_matching_threshold", stereo_matching_threshold_, 0.8);
    std::string model_name;
    nh_private_.param("model", model_name, std::string(""));

    ROS_INFO_STREAM("Settings:" << std::endl
        << "  model                      : " << model_name << std::endl
        << "  key_point_detector         : " << key_point_detector << std::endl
        << "  descriptor_extractor       : " << descriptor_extractor << std::endl
        << "  matching_threshold         : " << matching_threshold_ << std::endl
        << "  stereo_matching_threshold  : " << matching_threshold_ << std::endl
        << "  equalize_histogram         : " << equalize_histogram_ << std::endl
        << "  normalize_illumination     : " << normalize_illumination_ << std::endl
        << "  min_num_matches            : " << min_num_matches_ << std::endl);

    if (model_name != "")
    {
      loadModel(model_name);
    }
  }

  ~Features2D3DDetectorNode()
  {
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

      if (normalize_illumination_)
      {
        int k_size = image.cols/2;
        if (k_size % 2 == 0) k_size += 1;
        ROS_INFO("k_size = %i", k_size);
        cv::GaussianBlur(image, illumination_, cv::Size(k_size, k_size), 0);
        mean_illumination_ = cv::mean(image);
        image = image - illumination_ + mean_illumination_;
        cv::imwrite("/home/user/illumination.png", illumination_);
      }

      if (equalize_histogram_)
      {
        cv::Mat equalized_image;
        cv::equalizeHist(image, equalized_image);
        image = equalized_image;
      }

      std::vector<cv::KeyPoint> key_points;
      cv::Mat descriptors;
      key_point_detector_->detect(image, key_points);
      descriptor_extractor_->extract(image, key_points, descriptors);
      ROS_INFO("Training image has %zu features.", key_points.size());
      if (static_cast<int>(key_points.size()) >= min_model_features_count_)
      {
        model_.image = image;
        model_.key_points = key_points;
        model_.descriptors = descriptors;
        model_.outline.resize(training_request.outline.points.size());
        for (size_t i = 0; i < training_request.outline.points.size(); ++i)
        {
          model_.outline[i].x = training_request.outline.points[i].x;
          model_.outline[i].y = training_request.outline.points[i].y;
        }
        model_.origin.x = training_request.image_pose.x;
        model_.origin.y = training_request.image_pose.y;
        model_.theta = training_request.image_pose.theta;
        model_.object_id = training_request.object_id;
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

  void extractStereoFeatures(
      const cv::Mat& image_left, const cv::Mat& image_right,
      const sensor_msgs::CameraInfoConstPtr& l_camera_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_camera_info_msg,
      double matching_threshold,
      std::vector<cv::KeyPoint>& key_points,
      cv::Mat& descriptors, std::vector<cv::Point3f>& points3d)
  {
    key_points.clear();
    points3d.clear();
    descriptors = cv::Mat();
    std::vector<cv::KeyPoint> key_points_left;
    key_point_detector_->detect(image_left, key_points_left);
    cv::Mat descriptors_left;
    descriptor_extractor_->extract(image_left, key_points_left, descriptors_left);

    std::vector<cv::KeyPoint> key_points_right;
    key_point_detector_->detect(image_right, key_points_right);
    cv::Mat descriptors_right;
    descriptor_extractor_->extract(image_right, key_points_right, descriptors_right);

    // configure and perform matching
    feature_matching::StereoFeatureMatcher::Params params;
    params.max_y_diff = 2.0;
    params.max_angle_diff = 5.0;
    params.max_size_diff = 1;

    feature_matching::StereoFeatureMatcher matcher;
    matcher.setParams(params);
    std::vector<cv::DMatch> matches;
    matcher.match(key_points_left, descriptors_left, key_points_right,
                descriptors_right, matching_threshold, matches);

    // calculate 3D world points
    feature_matching::StereoDepthEstimator depth_estimator;
    depth_estimator.setCameraInfo(*l_camera_info_msg, *r_camera_info_msg);
    for (size_t i = 0; i < matches.size(); ++i)
    {
      int index_left = matches[i].queryIdx;
      int index_right = matches[i].trainIdx;
      cv::Point3d world_point;
      depth_estimator.calculate3DPoint(key_points_left[index_left].pt,
                                      key_points_right[index_right].pt,
                                      world_point);
      key_points.push_back(key_points_left[index_left]);
      descriptors.push_back(descriptors_left.row(index_left));
      points3d.push_back(world_point);
    }
  }
 
  virtual void detect(
     const sensor_msgs::ImageConstPtr& l_image_msg, 
     const sensor_msgs::ImageConstPtr& r_image_msg, 
     const sensor_msgs::CameraInfoConstPtr& l_camera_info_msg,
     const sensor_msgs::CameraInfoConstPtr& r_camera_info_msg,
     vision_msgs::DetectionArray& detection_array) 
  {
    if (model_.image.empty())
    {
      ROS_WARN_THROTTLE(10.0, "Detector has no model!");
      return;
    }
    cv::Mat image_left, image_right;
    cv_bridge::CvImageConstPtr l_cv_ptr, r_cv_ptr;
    try
    {
        l_cv_ptr = cv_bridge::toCvShare(l_image_msg, enc::MONO8);
        image_left = l_cv_ptr->image;
        r_cv_ptr = cv_bridge::toCvShare(r_image_msg, enc::MONO8);
        image_right = r_cv_ptr->image;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    if (normalize_illumination_)
    {
      image_right = image_right - illumination_ + mean_illumination_;
      image_left = image_left - illumination_ + mean_illumination_;
    }

    if (equalize_histogram_)
    {
      cv::Mat image_right_equalized;
      cv::equalizeHist(image_right, image_right_equalized);
      image_right = image_right_equalized;
      cv::Mat image_left_equalized;
      cv::equalizeHist(image_left, image_left_equalized);
      image_left = image_left_equalized;
    }

    // detect stereo features
    std::vector<cv::KeyPoint> key_points;
    cv::Mat descriptors;
    std::vector<cv::Point3f> points3d;
    extractStereoFeatures(image_left, image_right, 
        l_camera_info_msg, r_camera_info_msg, 
        stereo_matching_threshold_, key_points, descriptors, points3d);
    ROS_INFO("Detected %zu stereo features in current image.", points3d.size());

    // match with model
    cv::Mat match_mask;
    std::vector<cv::DMatch> matches;
    feature_matching::matching_methods::crossCheckThresholdMatching(descriptors, 
        model_.descriptors, matching_threshold_, match_mask, matches);

    std::vector<int> query_indices(matches.size()), train_indices(matches.size());
    std::vector<cv::Point3f> matched_points3d(matches.size());
    for (size_t i = 0; i < matches.size(); i++)
    {
      query_indices[i] = matches[i].queryIdx;
      train_indices[i] = matches[i].trainIdx;
      matched_points3d[i] = points3d[matches[i].queryIdx];
    }
    std::vector<cv::Point2f> current_points; 
    cv::KeyPoint::convert(key_points, current_points, query_indices);
    std::vector<cv::Point2f> model_points; 
    cv::KeyPoint::convert(model_.key_points, model_points, train_indices);
 
    bool detection_ok = false;
    cv::Mat inliers;
    cv::Mat homography;
    vision_msgs::Detection detection;
    detection.object_id = model_.object_id;
    detection.detector = "Features2D";
    if (static_cast<int>(matches.size()) >= min_num_matches_)
    {
      double reprojection_threshold = 2;
      homography = cv::findHomography(
          cv::Mat(model_points), cv::Mat(current_points), CV_RANSAC, reprojection_threshold, 
          inliers);
      if (!homography.empty())
      {
        ROS_INFO("Found homography with %i inliers.", cv::countNonZero(inliers));
      }
      cv::Mat r_vec, t_vec;
      const cv::Mat P(3,4, CV_64FC1, const_cast<double*>(l_camera_info_msg->P.data()));
      // We have to take K' here extracted from P to take the R|t into account
      // that was performed during rectification.
      // This way we obtain the pattern pose with respect to the same frame that
      // is used in stereo depth calculation.
      const cv::Mat camera_matrix = P.colRange(cv::Range(0,3));
      cv::Mat inliers_pnp;
      cv::Mat distortion(4, 1, CV_64FC1, cv::Scalar(0.0)); // no distortion
      assert(matched_points3d.size() == model_points.size());
      cv::solvePnPRansac(
          matched_points3d, model_points, camera_matrix, distortion, 
          r_vec, t_vec, false /* no extrinsic guess */,
          100 /* iterations */, 8.0 /* reproj. error */,
          100 /* min inliers */, inliers_pnp);
      int inliers_pnp_count = cv::countNonZero(inliers_pnp);
      ROS_INFO("Found camera pose with %i inliers", cv::countNonZero(inliers_pnp));
      if (inliers_pnp_count > 0)
      {
        tf::Vector3 axis(r_vec.at<double>(0, 0), r_vec.at<double>(1, 0), r_vec.at<double>(2, 0));
        double angle = cv::norm(r_vec);
        tf::Quaternion quaternion(axis, angle);
        tf::Vector3 translation(t_vec.at<double>(0, 0), t_vec.at<double>(1, 0), t_vec.at<double>(2, 0));
        tf::Transform transform(quaternion, translation);
        tf::poseTFToMsg(transform.inverse(), detection.training_pose);
        if (inliers_pnp_count >= 100)
        {
          detection.score = 1.0;
        }
        else
        {
          detection.score = inliers_pnp_count / 100.0;
        }
        detection_ok = true;
      }
    }

    if (features_image_pub_.getNumSubscribers() > 0)
    {
      cv::Mat canvas;
      // draw all matches
      cv::drawMatches(
          image_left, key_points, model_.image, model_.key_points, matches, canvas, 
          CV_RGB(255, 0, 0), CV_RGB(255, 0, 0), std::vector<char>(),
          cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      // draw inliers
      cv::drawMatches(
          image_left, key_points, model_.image, model_.key_points, matches, canvas, 
          CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), inliers,
          cv::DrawMatchesFlags::DRAW_OVER_OUTIMG /*| 
          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS*/);
      cv_bridge::CvImage cv_image;
      cv_image.header = l_image_msg->header;
      cv_image.encoding = enc::RGB8;
      cv_image.image = canvas;
      features_image_pub_.publish(cv_image.toImageMsg());
    }

    if (!homography.empty())
    {
      std::vector<cv::Point2f> original_points = model_.outline;
      original_points.push_back(model_.origin);
      cv::Point2f x_direction(model_.origin);
      x_direction.x += 100 * cos(model_.theta);
      x_direction.y += 100 * sin(model_.theta);
      original_points.push_back(x_direction);
      std::vector<cv::Point2f> transformed_points;
      cv::perspectiveTransform(original_points, transformed_points, homography);
      detection.outline.points.resize(transformed_points.size() - 2);
      for (size_t i = 0; i < transformed_points.size() - 2; ++i)
      {
        detection.outline.points[i].x = transformed_points[i].x;
        detection.outline.points[i].y = transformed_points[i].y;
        detection.outline.points[i].z = 0.0;
      }
      const cv::Point2f& transformed_origin = 
        transformed_points[transformed_points.size() - 2];
      const cv::Point2f& transformed_x_direction =
        transformed_points[transformed_points.size() - 1];
      detection.image_pose.x = transformed_origin.x;
      detection.image_pose.y = transformed_origin.y;
      cv::Point2f transformed_x_axis =
        transformed_x_direction - transformed_origin;
      detection.image_pose.theta = 
        atan2(transformed_x_axis.y, transformed_x_axis.x);
      double length = sqrt(transformed_x_axis.x * transformed_x_axis.x +
                           transformed_x_axis.y * transformed_x_axis.y);
      detection.scale = length / 100.0; // 100 was the untransformed length
      detection.homography.resize(homography.rows * homography.cols);
      assert(homography.type() == CV_64FC1);
      for (int i = 0; i < homography.rows; ++i)
        for (int j = 0; j < homography.cols; ++j)
          detection.homography[i * homography.cols + j] = 
            homography.at<double>(i, j);
    }

    if (detection_ok)
    {
      detection.header = l_image_msg->header;
      detection_array.detections.push_back(detection);
    }
    detection_array.header = l_image_msg->header;
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "features2d3d_object_detector");
  Features2D3DDetectorNode detector;
  ros::spin();
  return 0;
}

