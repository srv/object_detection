#include <boost/archive/text_iarchive.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "odat/serialization_support.h"
#include "odat/training_data.h"
#include "odat/exceptions.h"

#include "object_detection/feature_matching_detector.h"

object_detection::FeatureMatchingDetector::FeatureMatchingDetector(odat::ModelStorage::Ptr model_storage) : odat::Detector(model_storage)
{
}


void object_detection::FeatureMatchingDetector::detect()
{
  assert(camera_matrix_k_.data != NULL);
  detections_.clear();

  std::map<std::string, odat::FeatureSet3D>::const_iterator iter;
  for (iter = model_features_.begin(); iter != model_features_.end(); ++iter)
  {

    const odat::FeatureSet3D& model_features = iter->second;
    const std::string& model_name = iter->first;

    if (model_features.features_left.descriptor_name != features_.descriptor_name)
    {
      // ignore models that have other type of descriptor
      continue;
    }

    // perform matching and store matched image points and world points in separate structure
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> world_points;

    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = cv::DescriptorMatcher::create("BruteForce");
    std::vector<std::vector<cv::DMatch> > matches;
    // 1. query, 2. train
    int k = 2;
    descriptor_matcher->knnMatch(features_.descriptors, model_features.features_left.descriptors, matches, k);
    for (size_t m = 0; m < matches.size(); m++ )
    {
      if (matches[m].size() == 2)
      {
        float dist1 = matches[m][0].distance;
        float dist2 = matches[m][1].distance;
        static const float DISTANCE_RATIO_THRESHOLD = 0.8;
        if (dist1 / dist2 < DISTANCE_RATIO_THRESHOLD)
        {
          int queryIndex = matches[m][0].queryIdx;
          int trainIndex = matches[m][0].trainIdx;
          image_points.push_back(features_.key_points[queryIndex].pt);
          world_points.push_back(model_features.world_points[trainIndex]);
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
      int min_inliers = 100; // stop iteration if more inliers than this are found TODO make this model dependent
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

      odat::Detection detection;
      detection.label = model_name;
      detection.detector = getName();
      detection.score = 1.0; // TODO add score based on number of matchings (?)
      detection.transform = transform;
      detections_.push_back(detection);
    }
  }
}

void object_detection::FeatureMatchingDetector::loadModels(const std::vector<std::string>& models)
{
  for (size_t i = 0; i < models.size(); ++i) {
    std::string model_blob;
    model_storage_->loadModel(models[i], getName(), model_blob);
    std::istringstream istr(model_blob);
    boost::archive::text_iarchive archive(istr);
    archive >> model_features_[models[i]];
  }
}

std::vector<std::string> object_detection::FeatureMatchingDetector::getLoadedModels() const
{
  std::vector<std::string> loaded_models;
  std::map<std::string, odat::FeatureSet3D>::const_iterator iter;
  for (iter = model_features_.begin(); iter != model_features_.end(); ++iter)
  {
    loaded_models.push_back(iter->first);
  }
  return loaded_models;
}

void object_detection::FeatureMatchingDetector::saveModel(const std::string& model)
{
  assert(model_features_.find(model) != model_features_.end());
  std::ostringstream ostr;
  boost::archive::text_oarchive archive(ostr);
  archive << model_features_[model];
  model_storage_->saveModel(model, getName(), ostr.str());
}

void object_detection::FeatureMatchingDetector::startTraining(const std::string& /*name*/)
{
  // nothing to do here
}

void object_detection::FeatureMatchingDetector::trainInstance(const std::string& name, const odat::TrainingData& data)
{
  // we do not support multiple training data, old data is overwritten
  if (data.features_3d.world_points.size() < 4)
  {
    throw odat::Exception("Insufficient training data for " + getName());
  }
  model_features_[name] = data.features_3d;
}

void object_detection::FeatureMatchingDetector::endTraining(const std::string& /*name*/)
{
  // nothing to do here
}

