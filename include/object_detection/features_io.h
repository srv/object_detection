
namespace object_detection
{

namespace features_io
{

bool loadFeatures(const std::string& filename, std::vector<cv::Point3f>& points,
    cv::Mat& descriptors)
{
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) return false;
  cv::FileNode fs_points = fs["Feature_Locations"];
  points.clear();
  for (cv::FileNodeIterator it = fs_points.begin(); it != fs_points.end(); ++it)
  {
    cv::Point3f cv_point;
    (*it)["x"] >> cv_point.x;
    (*it)["y"] >> cv_point.y;
    (*it)["z"] >> cv_point.z;
    points.push_back(cv_point);
  }
  fs["Feature_Descriptors"] >> descriptors;
  fs.release();
  return true;
}

}

}

