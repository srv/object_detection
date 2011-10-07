#include <boost/archive/text_iarchive.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "odat/serialization_support.h"
#include "odat/training_data.h"
#include "odat/exceptions.h"

#include "object_detection/histogram_utilities.h"
#include "object_detection/shape_detector.h"
#include "object_detection/shape_processing.h"
#include "object_detection/shape_matching.h"

const float object_detection::ShapeDetector::DEFAULT_MATCHING_SCORE_THRESHOLD = 0.3;
const float object_detection::ShapeDetector::DEFAULT_MIN_SCALE = 0.2;
const float object_detection::ShapeDetector::DEFAULT_MAX_SCALE = 5.0;

object_detection::ShapeDetector::ShapeDetector(odat::ModelStorage::Ptr model_storage) :
        odat::Detector(model_storage),
        matching_score_threshold_(DEFAULT_MATCHING_SCORE_THRESHOLD),
        min_scale_(DEFAULT_MIN_SCALE),
        max_scale_(DEFAULT_MAX_SCALE)
{
}

void object_detection::ShapeDetector::detect()
{
  detections_.clear();

  //check for input
  if (input_detections_.size() == 0)
  {
    throw odat::Exception("Insufficient data for detection in " + getName());
  }

  for (size_t i = 0; i < input_detections_.size(); ++i)
  {
    const cv::Mat& candidate_mask = input_detections_[i].mask.mask;
    if (candidate_mask.empty())
    {
      // skip detections without binary mask
      continue;
    }

    std::vector<shape_processing::Shape> detected_shapes = shape_processing::extractShapes(candidate_mask);
    std::vector<shape_processing::Shape> biggest_shapes = shape_processing::getBiggestShapes(detected_shapes);
    const std::vector<cv::Point> candidate_shape = biggest_shapes[0];

    std::map<std::string, std::vector<cv::Point> >::const_iterator iter;
    for (iter = model_shapes_.begin(); iter != model_shapes_.end(); ++iter)
    {
      const std::string& model_name = iter->first;
      const std::vector<cv::Point>& model_shape = iter->second;
          // get shape of mask
      double score;
      ShapeMatching::MatchingParameters matching_parameters = ShapeMatching::matchShapes(candidate_shape, model_shape, &score);

      if (score > matching_score_threshold_ &&
          matching_parameters.scale >= min_scale_ &&
          matching_parameters.scale <= max_scale_)
      {
        // report detection
        odat::Detection detection;
        detection.mask = input_detections_[i].mask;
        //detection.mask.roi = shape_processing::boundingRect(candidate_shape);
        //detection.mask.mask = shape_processing::minimalMask(candidate_shape);
        detection.label = model_name;
        detection.detector = getName();
        detection.score = score;
        detection.scale = matching_parameters.scale;
        // TODO insert pose information!
        detection.transform.create(2, 3, CV_32F);
        detection.transform.at<float>(0, 2) = matching_parameters.shift_x;
        detection.transform.at<float>(1, 2) = matching_parameters.shift_y;
        detections_.push_back(detection);
      }
    }
  }
}

void object_detection::ShapeDetector::loadModels(const std::vector<std::string>& models)
{
  for (size_t i = 0; i < models.size(); ++i) {
    std::string model_blob;
    model_storage_->loadModel(models[i], getName(), model_blob);
    std::istringstream istr(model_blob);
    boost::archive::text_iarchive archive(istr);
    std::vector<cv::Point> shape;
    archive >> shape;
    model_shapes_[models[i]] = shape;
  }
}

std::vector<std::string> object_detection::ShapeDetector::getLoadedModels() const
{
  std::vector<std::string> loaded_models;
  std::map<std::string, std::vector<cv::Point> >::const_iterator iter;
  for (iter = model_shapes_.begin(); iter != model_shapes_.end(); ++iter)
  {
    loaded_models.push_back(iter->first);
  }
  return loaded_models;
}

void object_detection::ShapeDetector::saveModel(const std::string& model)
{
  assert(model_shapes_.find(model) != model_shapes_.end());
  std::ostringstream ostr;
  boost::archive::text_oarchive archive(ostr);
  archive << model_shapes_[model];
  model_storage_->saveModel(model, getName(), ostr.str());
}

void object_detection::ShapeDetector::startTraining(const std::string& name)
{
  // nothing to do here
}

void object_detection::ShapeDetector::trainInstance(const std::string& name, const odat::TrainingData& data)
{
  if (data.mask.mask.empty())
  {
    throw odat::Exception("Insufficient training data for " + getName());
  }
  std::vector<shape_processing::Shape> shapes = shape_processing::extractShapes(data.mask.mask);
  std::vector<shape_processing::Shape> biggest_shapes = shape_processing::getBiggestShapes(shapes);
  // save biggest shape as model
  model_shapes_[name] = shape_processing::shift(biggest_shapes[0], -data.mask.roi.width / 2, -data.mask.roi.height / 2);
}

void object_detection::ShapeDetector::endTraining(const std::string& name)
{
  saveModel(name);
}

