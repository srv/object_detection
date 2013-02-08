#include <string>
#include <ros/package.h>

/**
 * Base class for all detector nodes.
 */
class Detector
{
public:
  virtual ~Detector() {}

  /**
   * @return name of the directory where the detector models are stored.
   */
  static std::string getModelDir()
  {
    return ros::package::getPath(ROS_PACKAGE_NAME) + "/models/";
  }
};

