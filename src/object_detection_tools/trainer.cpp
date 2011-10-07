#include <iostream>
#include <ros/package.h>
#include <boost/program_options.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <odat/fs_model_storage.h>

#include "object_detection/color_detector.h"
#include "object_detection/shape_detector.h"
#include "object_detection/feature_matching_detector.h"


namespace po = boost::program_options;

int main(int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("image,I", po::value<std::string>(), "image file showing the object")
    ("mask,M", po::value<std::string>(), "mask file (binary image) marking the object")
    ("stereo_features,S", po::value<std::string>(), "stereo features file (PCD)")
    ("detector,R", po::value<std::string>()->required(), "name of the detector to train")
    ("name,N", po::value<std::string>()->required(), "name for the object to train")
    ("db_type,D", po::value<std::string>()->default_value("filesystem"), "database type")
    ("connection_string,C", po::value<std::string>()->default_value(ros::package::getPath("object_detection") + "/models"), "database connection string")
    ;
  po::variables_map vm;
  try
  {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);    
  } catch (const po::error& error)
  {
      std::cerr << "Error parsing program options: " << std::endl;
      std::cerr << "  " << error.what() << std::endl;
      std::cerr << desc << std::endl;
      return -1;
  }

  odat::TrainingData training_data;
  if (vm.count("image"))
  {
    training_data.image = cv::imread(vm["image"].as<std::string>());
    if (training_data.image.data == NULL)
    {
      std::cerr << "Cannot load image " << vm["image"].as<std::string>() << "!" << std::endl;
      return -2;
    }
  }

  if (vm.count("mask"))
  {
    training_data.mask.mask = cv::imread(vm["mask"].as<std::string>(), 0); // 0 = load as greyscale
    if (training_data.mask.mask.data == NULL)
    {
      std::cerr << "Cannot load mask " << vm["mask"].as<std::string>() << "!" << std::endl;
      return -3;
    }
    // we assume that the mask image has same size as image
    training_data.mask.roi = cv::Rect(0, 0, training_data.image.cols, training_data.image.rows);
  }

  if (vm.count("stereo_features"))
  {
    std::cerr << "training with stereo features not supported yet" << std::endl;
    return -6;
  }

  std::string object_name = vm["name"].as<std::string>();
  std::string db_type = vm["db_type"].as<std::string>();
  std::string connection_string = vm["connection_string"].as<std::string>();

  odat::ModelStorage::Ptr model_storage;
  if (db_type=="filesystem") {
      model_storage = boost::make_shared<odat::FilesystemModelStorage>(connection_string);
  }
  else {
    std::cerr << "Unknown model storage type" << std::endl;
    return -5;
  }

  std::string detector = vm["detector"].as<std::string>();

  odat::Trainable::Ptr trainable;
  if (detector == "ColorDetector")
  {
    trainable.reset(new object_detection::ColorDetector(model_storage));
  }
  else if (detector == "ShapeDetector")
  {
    trainable.reset(new object_detection::ShapeDetector(model_storage));
  }
  else if (detector == "FeatureMatchingDetector")
  {
    trainable.reset(new object_detection::FeatureMatchingDetector(model_storage));
  }
  else
  {
    std::cerr << "Don't know how to train '" << detector << "'." << std::endl;
    return -6;
  }
  trainable->startTraining(object_name);
  trainable->trainInstance(object_name, training_data);
  trainable->endTraining(object_name);

  std::cout << "Model '" << object_name << "' for " << detector << " trained." << std::endl;

  return 0;
}

