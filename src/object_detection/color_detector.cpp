#include <boost/archive/text_iarchive.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "odat/serialization_support.h"
#include "odat/training_data.h"
#include "odat/exceptions.h"

#include "object_detection/histogram_utilities.h"
#include "object_detection/color_detector.h"
#include "object_detection/shape_processing.h"
/*

// ------------- cv::MatND

namespace boost 
{ 
  namespace serialization 
  {

    template<class Archive>
    void save(Archive & ar, const cv::MatND& mat, const unsigned int version)
    {
      cv::MatND mat_;
      mat_ = mat;
      if (!mat.isContinuous()) {
        mat_ = mat.clone();
      }
      ar & mat_.dims;
      for (int i = 0; i < mat_.dims; ++i)
      {
        ar & mat_.size[i];
      }
      int type = mat_.type();
      ar & type;
      ar & boost::serialization::make_binary_object(mat_.data, mat_.step[mat_.dims - 1] * mat_.size[mat_dims - 1]);
    }

    template<class Archive>
    void load(Archive & ar, cv::MatND& mat, const unsigned int version)
    {
      int dims, type;
      ar & dims;
      int* sizes = new int[dims];
      for (int i = 0; i < dims; ++i)
        ar & sizes[i];
      ar & type;
      mat.create(dims, sizes, type);
      ar & boost::serialization::make_binary_object(mat.data, mat.step[mat_.dims - 1] * mat.size[mat_dims - 1]);
      delete sizes;
    }
  }
}
BOOST_SERIALIZATION_SPLIT_FREE(cv::MatND);
*/

object_detection::ColorDetector::ColorDetector(odat::ModelStorage::Ptr model_storage) : odat::Detector(model_storage),
  num_hue_bins_(DEFAULT_NUM_HUE_BINS),
  num_saturation_bins_(DEFAULT_NUM_SATURATION_BINS),
  min_saturation_(DEFAULT_MIN_SATURATION),
  min_value_(DEFAULT_MIN_VALUE),
  morph_element_size_(DEFAULT_MORPH_ELEMENT_SIZE),
  show_images_(false)
{
}

// adapt histogram according to settings
cv::MatND object_detection::ColorDetector::adaptHistogram(const std::string& model_name, const cv::MatND& model_histogram)
{
  cv::MatND adapted_histogram;
  cv::resize(model_histogram, adapted_histogram, cv::Size(num_hue_bins_, num_saturation_bins_));

  // set small saturations to zero
  int black_sat_bins = min_saturation_ * num_saturation_bins_ / 256;
  for (int s = 0; s <= black_sat_bins; ++s)
  {
    adapted_histogram.col(s) = cv::Scalar::all(0);
  }

  if (show_images_)
  {
    histogram_utilities::showHSHistogram(adapted_histogram, model_name + " adapted model histogram");
    cv::waitKey(5);
  }
  return adapted_histogram;
}

void object_detection::ColorDetector::detect()
{
  detections_.clear();

  //check for input
  if (image_.empty())
  {
    throw odat::Exception("Insufficient data for detection in " + getName());
  }

  std::map<std::string, cv::MatND>::const_iterator iter;
  for (iter = model_histograms_.begin(); iter != model_histograms_.end(); ++iter)
  {
    const std::string& model_name = iter->first;
    cv::MatND model_histogram = adaptHistogram(model_name, iter->second);
    cv::Mat hsv_image;
    cv::cvtColor(image_, hsv_image, CV_BGR2HSV);
    cv::Mat backprojection = histogram_utilities::calculateBackprojection(model_histogram, hsv_image);
    cv::Mat binary;
    cv::threshold(backprojection, binary, 127, 255, CV_THRESH_BINARY);

    // some opening
    int element_size = morph_element_size_;
    cv::Mat element = cv::Mat::zeros(element_size, element_size, CV_8UC1);
    cv::circle(element, cv::Point(element_size / 2, element_size / 2), element_size / 2, cv::Scalar(255), -1);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, element);

    if (show_images_)
    {
      cv::namedWindow(model_name +"-backprojection", 0);
      cv::namedWindow(model_name + "-backprojection-thresholded-opened", 0);
      cv::imshow(model_name + "-backprojection", backprojection);
      cv::imshow(model_name + "-backprojection-thresholded-opened", binary);
      cv::waitKey(5);
    }

    std::vector<shape_processing::Shape> detected_shapes = shape_processing::extractShapes(binary);
    std::vector<shape_processing::Shape> biggest_shapes = shape_processing::getBiggestShapes(detected_shapes);
    for (size_t i = 0; i < biggest_shapes.size(); ++i)
    {
      // report detection
      odat::Detection detection;
      detection.mask.roi = shape_processing::boundingRect(biggest_shapes[i]);
      detection.mask.mask = shape_processing::minimalMask(biggest_shapes[i]);
      detection.label = model_name;
      detection.detector = getName();
      detection.score = 1;
      detections_.push_back(detection);
    }
  }
}

void object_detection::ColorDetector::loadModels(const std::vector<std::string>& models)
{
  for (size_t i = 0; i < models.size(); ++i) {
    std::string model_blob;
    model_storage_->loadModel(models[i], getName(), model_blob);
    std::istringstream istr(model_blob);
    boost::archive::text_iarchive archive(istr);
    cv::MatND histogram;
    archive >> histogram;
    model_histograms_[models[i]] = histogram;
  }
}

std::vector<std::string> object_detection::ColorDetector::getLoadedModels() const
{
  std::vector<std::string> loaded_models;
  std::map<std::string, cv::MatND>::const_iterator iter;
  for (iter = model_histograms_.begin(); iter != model_histograms_.end(); ++iter)
  {
    loaded_models.push_back(iter->first);
  }
  return loaded_models;
}

void object_detection::ColorDetector::saveModel(const std::string& model)
{
  assert(model_histograms_.find(model) != model_histograms_.end());
  std::ostringstream ostr;
  boost::archive::text_oarchive archive(ostr);
  archive << model_histograms_[model];
  model_storage_->saveModel(model, getName(), ostr.str());
}

void object_detection::ColorDetector::startTraining(const std::string& name)
{
  // nothing to do here
}

void object_detection::ColorDetector::trainInstance(const std::string& name, const odat::TrainingData& data)
{
  training_data_[name].push_back(data);
}

void object_detection::ColorDetector::endTraining(const std::string& name)
{
  int num_hue_bins_ = 180;
  int num_saturation_bins_ = 256;
  assert(training_data_.find(name) != training_data_.end());
  cv::MatND image_histogram;
  cv::MatND object_histogram;
  const std::vector<odat::TrainingData>& training_data = training_data_[name];
  for (size_t i = 0; i < training_data.size(); ++i)
  {
    cv::Mat object_mask = cv::Mat::zeros(training_data[i].image.rows, training_data[i].image.cols, CV_8U);
    cv::Mat local_object_mask = object_mask(training_data[i].mask.roi);
    training_data[i].mask.mask.copyTo(local_object_mask);
    assert(training_data[i].image.type() == CV_8UC3);
    cv::Mat hsv_image;
    cv::cvtColor(training_data[i].image, hsv_image, CV_BGR2HSV);
    if (i == 0)
    {
      image_histogram = histogram_utilities::calculateHistogram(hsv_image, num_hue_bins_, num_saturation_bins_, cv::Mat());
      object_histogram = histogram_utilities::calculateHistogram(hsv_image, num_hue_bins_, num_saturation_bins_, object_mask);
    }
    else
    {
      histogram_utilities::accumulateHistogram(hsv_image, num_hue_bins_, num_saturation_bins_, cv::Mat(), image_histogram);
      histogram_utilities::accumulateHistogram(hsv_image, num_hue_bins_, num_saturation_bins_, object_mask, object_histogram);
    }
  }
  cv::MatND model_histogram = object_histogram / image_histogram * 255;

  if (show_images_)
  {
    histogram_utilities::showHSHistogram(model_histogram, "model histogram");
  }
  model_histograms_[name] = model_histogram;
  training_data_.erase(name);
  saveModel(name);
}

