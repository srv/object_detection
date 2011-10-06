#include <iostream>
#include <ros/package.h>
#include <boost/program_options.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <odat/fs_model_storage.h>

#include "object_detection/color_detector.h"
#include "object_detection/feature_matching_detector.h"


namespace po = boost::program_options;

int main(int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("image,I", po::value<std::string>(), "image file showing the object")
    ("mask,M", po::value<std::string>(), "mask file (binary image) marking the region of interest")
    ("features3d,S", po::value<std::string>(), "features file (PCD)")
    ("detector,R", po::value<std::string>()->required(), "detector to run")
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

  odat::ModelStorage::Ptr model_storage;
  if (vm["db_type"].as<std::string>() == "filesystem") {
      model_storage = boost::make_shared<odat::FilesystemModelStorage>(vm["connection_string"].as<std::string>());
  }
  else {
    std::cerr << "Unknown model storage type" << std::endl;
    return -5;
  }

  odat::Detector::Ptr detector;
  std::string detector_name = vm["detector"].as<std::string>();
  if (detector_name == "ColorDetector")
  {
    detector.reset(new object_detection::ColorDetector(model_storage));
  }
  else if (detector_name == "FeatureMatchingDetector")
  {
    detector.reset(new object_detection::FeatureMatchingDetector(model_storage));
  }
  else
  {
    std::cerr << "Don't know how to instantiate '" << detector_name << "'." << std::endl;
    return -6;
  }

  if (vm.count("image"))
  {
    cv::Mat image = cv::imread(vm["image"].as<std::string>());
    if (image.data == NULL)
    {
      std::cerr << "Cannot load image " << vm["image"].as<std::string>() << "!" << std::endl;
      return -2;
    }
    detector->setImage(image);
  }

  if (vm.count("mask"))
  {
    odat::Mask mask;
    mask.mask = cv::imread(vm["mask"].as<std::string>(), 0); // 0 = load as greyscale
    if (mask.mask.data == NULL)
    {
      std::cerr << "Cannot load mask " << vm["mask"].as<std::string>() << "!" << std::endl;
      return -3;
    }
    mask.roi = cv::Rect(0, 0, mask.mask.cols, mask.mask.rows);
    std::vector<odat::Mask> masks;
    masks.push_back(mask);
    detector->setMasks(masks);
  }

  if (vm.count("features"))
  {
    std::cerr << "Feature setting not yet supported" << std::endl;
    return -4;
  }

  std::string db_type = vm["db_type"].as<std::string>();
  std::string connection_string = vm["connection_string"].as<std::string>();

  detector->loadAllModels();
  detector->detect();

  std::vector<odat::Detection> detections = detector->getDetections();
  std::cout << detections.size() << " detections: " << std::endl;
  for (size_t i = 0; i < detections.size(); ++i)
  {
    std::cout << "#" << i << ":" << std::endl;
    std::cout << detections[i] << std::endl;
    std::cout << std::endl;
    if (!detections[i].mask.mask.empty())
    {
      std::ostringstream name;
      name << "detection-" << i << "-" << detector->getName() << "-mask.jpg";
      cv::imwrite(name.str(), detections[i].mask.mask);
    }
  }

  return 0;
}

