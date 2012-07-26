#include <iostream>

#include <opencv2/core/core.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "object_detection/features_io.h"

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cout << "USAGE: " << argv[0] << " FEATURES_FILE PCD_FILE" << std::endl;
    return 0;
  }
  std::string features_filename(argv[1]);
  std::string pcd_filename(argv[2]);

  std::vector<cv::Point3f> points;
  cv::Mat features;
  object_detection::features_io::loadFeatures(
      features_filename, points, features);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (size_t i = 0; i < points.size(); ++i)
  {
    pcl::PointXYZ point;
    point.x = points[i].x; 
    point.y = points[i].y;
    point.z = points[i].z;
    cloud.push_back(point);
  }

  return pcl::io::savePCDFile(pcd_filename, cloud);
}



