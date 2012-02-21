#include <iostream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "object_detection/model3d_alignment.h"
#include "object_detection/model3d_fusion.h"
#include "object_detection/pcl_descriptor.h"


namespace po = boost::program_options;
using object_detection::PclDescriptor;

std::vector<std::string> getModelList(const std::string& dir_name)
{
  std::vector<std::string> model_list;
  using namespace boost::filesystem;
  using namespace boost::algorithm;
  path dir_path(dir_name);
  directory_iterator end;
  std::vector<std::string> point_cloud_files;
  for (directory_iterator iter(dir_path); iter != end; ++iter)
  {
    if (is_regular_file(*iter))
    {
      if (starts_with(iter->path().filename().string(), "points"))
      {
        const std::string descriptors_filename = replace_last_copy(iter->path().string(), "points", "descriptors");
        path descriptors_path(descriptors_filename);
        if (is_regular_file(descriptors_path))
        {
          std::string model_name = iter->path().filename().string().substr(6);
          model_name.resize(model_name.length() - 4); // get rid of ".pcd"
          model_list.push_back(model_name);
          std::cout << "found model " << model_name << std::endl;
        }
      }
    }
  }

  std::sort(model_list.begin(), model_list.end());
  return model_list;
}

typedef object_detection::Model3D<pcl::PointXYZ, object_detection::PclDescriptor> Model;

Model::Ptr loadModel(const std::string& dir_name, const std::string& model_name)
{
  std::string points_file = dir_name + "/points" + model_name + ".pcd";
  std::string descriptors_file = dir_name + "/descriptors" + model_name + ".pcd";
  pcl::PointCloud<pcl::PointXYZ> point_cloud;
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(points_file, point_cloud) == -1)
  {
    std::cerr << "Couldn't read file '" << points_file << "'" << std::endl;
  }
  std::cout << "Loaded " << point_cloud.points.size() << " points from " 
            << points_file << "." << std::endl;

  pcl::PointCloud<PclDescriptor> descriptor_cloud;
  if (pcl::io::loadPCDFile<PclDescriptor>(descriptors_file, descriptor_cloud) == -1)
  {
    std::cerr << "Couldn't read file '" << descriptors_file << "'" << std::endl;
  }
  std::cerr << "Loaded " << descriptor_cloud.points.size() 
            << " descriptors from " << descriptors_file << "." << std::endl;

  assert(point_cloud.points.size() == descriptor_cloud.points.size());
  Model::Ptr model(new Model());
  model->setName(model_name);
  for (size_t i = 0; i < point_cloud.points.size(); ++i)
  {
    std::vector<object_detection::PclDescriptor> descriptors(1);
    descriptors[0] = descriptor_cloud.points[i];
    model->addNewPoint(point_cloud.points[i], descriptors);
  }
  return model;
}

std::vector<Model::Ptr> loadModels(const std::string& dir_name)
{
  std::vector<std::string> model_list = getModelList(dir_name);
  std::vector<Model::Ptr> models;
  for (size_t i = 0; i < model_list.size(); ++i)
  {
    models.push_back(loadModel(dir_name, model_list[i]));
  }
  return models;
}

int main(int argc, char** argv)
{
  srand(time(0));
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("data_dir,D", po::value<std::string>()->required(), "Directory with input data")
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

  std::string data_dir = vm["data_dir"].as<std::string>();

  std::vector<Model::Ptr> models = loadModels(data_dir);
  if (models.size() == 0)
  {
    std::cerr << "No models found in " << data_dir << "." << std::endl;
    return -2;
  }
  else
  {
    std::cout << "Loaded " << models.size() << " models." << std::endl;
  }

  /*
  pcl::visualization::PCLVisualizer viewer("3D Model");
  viewer.setBackgroundColor(0, 0, 0);
  for (size_t i = 0; i < models.size(); ++i)
  {
    Model::PointCloudConstPtr cloud = models[i]->getPointCloud();
    std::string name = models[i]->getName();
    pcl::visualization::PointCloudColorHandlerRandom<Model::PointType> color_handler(cloud);
    viewer.addPointCloud<Model::PointType>(cloud, color_handler, name);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, name);
    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();
  }
  while (!viewer.wasStopped())
  {
    viewer.spinOnce(100);
  }
  */

  return 0;
}

