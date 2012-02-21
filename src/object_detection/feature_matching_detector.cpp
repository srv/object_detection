#include <opencv2/calib3d/calib3d.hpp>

#include "object_detection/feature_matching_detector.h"

// init default parameters
object_detection::FeatureMatchingDetector::Params::Params() :
  descriptor_matcher("BruteForce"),
  distance_ratio_threshold(0.8)
{
}

object_detection::FeatureMatchingDetector::FeatureMatchingDetector()
{
}

void object_detection::FeatureMatchingDetector::detect(const odat::FeatureSet& features)
{
  assert(camera_matrix_k_.data != NULL);

  // perform matching and store matched image points and world points in separate structure
  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3d> world_points;

  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = cv::DescriptorMatcher::create(params_.descriptor_matcher);
  assert(descriptor_matcher != NULL);

  std::vector<std::vector<cv::DMatch> > matches;
  // 1. query, 2. train
  int k = 2;
  descriptor_matcher->knnMatch(features.descriptors, model_features_.features_left.descriptors, matches, k);
  for (size_t m = 0; m < matches.size(); m++ )
  {
    if (matches[m].size() == 2)
    {
      float dist1 = matches[m][0].distance;
      float dist2 = matches[m][1].distance;
      if (dist1 / dist2 < params_.distance_ratio_threshold)
      {
        int queryIndex = matches[m][0].queryIdx;
        int trainIndex = matches[m][0].trainIdx;
        image_points.push_back(features.key_points[queryIndex].pt);
        unsigned int world_point_index = model_features_.descriptor_index_to_world_point_index[trainIndex];
        assert(world_point_index < model_features_.world_points.size());
        world_points.push_back(model_features_.world_points[world_point_index]);
      }
    }
  }

  if (world_points.size() > 3)
  {
    // matching done, calculate transform
    cv::Mat distortion; // no distortion, we assume our features are extracted on rectified images
    cv::Mat r_vec(3, 1, CV_64FC1);
    cv::Mat t_vec(3, 1, CV_64FC1);
    bool use_extrinsic_guess = false;
    int num_iterations = 100;
    float allowed_reprojection_error = 8.0; // used by ransac to classify inliers
    int min_inliers = model_features_.world_points.size() / 2; // stop iteration if more inliers than this are found
    if (min_inliers < 3) min_inliers = 3;
    cv::Mat inliers;
    cv::solvePnPRansac(world_points, image_points, camera_matrix_k_, distortion, 
        r_vec, t_vec, use_extrinsic_guess, num_iterations, allowed_reprojection_error, min_inliers, inliers);
    cv::Mat r_mat;
    cv::Rodrigues(r_vec, r_mat);
    cv::Mat transform(3, 4, CV_64FC1);
    cv::Mat transform_r = transform.colRange(cv::Range(0, 3));
    cv::Mat transform_t = transform.col(4);
    r_mat.copyTo(transform_r);
    t_vec.copyTo(transform_t);

    /*
    odat::Detection detection;
    detection.label = model_name;
    detection.detector = getName();
    detection.score = 1.0; // TODO add score based on number of matchings (?)
    detection.pose = transform;
    detections_.push_back(detection);
    */
  }
}

