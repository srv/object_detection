#include <iostream>
#include <ros/package.h>
#include <boost/program_options.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <odat/fs_model_storage.h>

#include "object_detection/color_detector.h"


namespace po = boost::program_options;

int main(int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("image,I", po::value<std::string>()->required(), "image file showing the object")
    ("mask,M", po::value<std::string>()->required(), "mask file (binary image) marking the object")
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
  std::string image_file = vm["image"].as<std::string>();
  std::string mask_file = vm["mask"].as<std::string>();
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

  odat::TrainingData training_data;
  training_data.image = cv::imread(image_file);
  if (training_data.image.data == NULL)
  {
    std::cerr << "Cannot load image " << image_file << "!" << std::endl;
    return -2;
  }
  training_data.mask.mask = cv::imread(mask_file, 0); // 0 = load as greyscale
  if (training_data.mask.mask.data == NULL)
  {
    std::cerr << "Cannot load mask " << mask_file << "!" << std::endl;
    return -3;
  }
  training_data.mask.roi = cv::Rect(0, 0, training_data.image.cols, training_data.image.rows);

  if (training_data.mask.mask.rows != training_data.image.rows ||
      training_data.mask.mask.cols != training_data.image.cols)
  {
    std::cerr << "Image and mask have to be of same size!" << std::endl;
    return -4;
  }

  object_detection::ColorDetector detector(model_storage);
  detector.startTraining(object_name);
  detector.trainInstance(object_name, training_data);
  detector.endTraining(object_name);

  detector.setImage(training_data.image);
  detector.detect();

  return 0;
}

