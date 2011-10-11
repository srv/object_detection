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
    ("image", po::value<std::string>()->required(), "image file showing the object")
    ("mask", po::value<std::string>()->required(), "mask file (binary image of same size as image) marking the object")
//    ("stereo_features,S", po::value<std::string>(), "stereo features file (PCD)")
    ("name", po::value<std::string>()->required(), "name for the object to train")
    ("origin_x", po::value<double>()->required(), "x position of the origin of the object in the image, given in pixels")
    ("origin_y", po::value<double>()->required(), "y position of the origin of the object in the image, given in pixels")
    ("origin_theta", po::value<double>()->required(), "angle of the origin of the object in the image, given in radiants, positive values turn from x axis to y axis")
    ("db_type", po::value<std::string>()->default_value("filesystem"), "database type")
    ("connection_string", po::value<std::string>()->default_value(ros::package::getPath("object_detection") + "/models"), "database connection string")
    ;
  po::variables_map vm;
  try
  {
      po::store(po::parse_command_line(argc, argv, desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);
      //po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);    
  } catch (const po::error& error)
  {
      std::cerr << "Error parsing program options: " << std::endl;
      std::cerr << "  " << error.what() << std::endl;
      std::cerr << desc << std::endl;
      return -1;
  }

  odat::TrainingData training_data;
  training_data.image = cv::imread(vm["image"].as<std::string>());
  if (training_data.image.data == NULL)
  {
    std::cerr << "Cannot load image " << vm["image"].as<std::string>() << "!" << std::endl;
    return -2;
  }

  training_data.mask.mask = cv::imread(vm["mask"].as<std::string>(), 0); // 0 = load as greyscale
  if (training_data.mask.mask.data == NULL)
  {
    std::cerr << "Cannot load mask " << vm["mask"].as<std::string>() << "!" << std::endl;
    return -3;
  }
  
  // check that the mask image has same size as image
  training_data.mask.roi = cv::Rect(0, 0, training_data.image.cols, training_data.image.rows);
  if (training_data.image.cols != training_data.mask.mask.cols ||
      training_data.image.rows != training_data.mask.mask.rows)
  {
    std::cerr << "Image and mask are not of same size!" << std::endl;
    return -4;
  }

  training_data.image_pose.x = vm["origin_x"].as<double>();
  training_data.image_pose.y = vm["origin_y"].as<double>();
  training_data.image_pose.theta = vm["origin_theta"].as<double>();

  /*
  if (vm.count("stereo_features"))
  {
    std::cerr << "training with stereo features not supported yet" << std::endl;
    return -6;
  }
  */

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

  // train color detector
  object_detection::ColorDetector color_detector(model_storage);
  color_detector.setShowImages(true);
  color_detector.startTraining(object_name);
  color_detector.trainInstance(object_name, training_data);
  color_detector.endTraining(object_name); // this saves the model

  // run color detector and store its output to train shape detector
  color_detector.setImage(training_data.image);
  color_detector.detect();
  std::vector<odat::Detection> detections = color_detector.getDetections();
  if (detections.size() < 1)
  {
    std::cout << "Error during color detection: " << detections.size() << " detections made." << std::endl;
    return -6;
  }
  // train shape detector
  training_data.mask = detections[0].mask;
  object_detection::ShapeDetector shape_detector(model_storage);
  shape_detector.startTraining(object_name);
  shape_detector.trainInstance(object_name, training_data);
  shape_detector.endTraining(object_name);

  std::cout << "Model '" << object_name << "' trained." << std::endl;

  cv::namedWindow("shape model");
  cv::imshow("shape model", training_data.mask.mask);

  std::cout << "Press any key to quit." << std::endl;
  cv::waitKey();

  return 0;
}

