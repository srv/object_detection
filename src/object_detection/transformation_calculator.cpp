
#include <opencv2/core/core.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>
#include <pcl/features/feature.h>

#include "transformation_calculator.h"

namespace object_detection
{

// converts vector of OpenCV points to point cloud
pcl::PointCloud<pcl::PointXYZ> toPcl_(const std::vector<cv::Point3f>& points)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (size_t i = 0; i < points.size(); ++i)
    {
        pcl::PointXYZ point;
        point.x = points[i].x;
        point.y = points[i].y;
        point.z = points[i].z;
        cloud.points.push_back(point);
    }
    return cloud;
}

void TransformationCalculator::calculateRigidTransformation(
        const std::vector<cv::Point3f>& points1,
        const std::vector<cv::Point3f>& points2, cv::Mat& transform)
{
    pcl::PointCloud<pcl::PointXYZ> cloud1 = toPcl_(points1);
    pcl::PointCloud<pcl::PointXYZ> cloud2 = toPcl_(points2);
    Eigen::Matrix4f transformation;
    pcl::estimateRigidTransformationSVD(cloud1, cloud2, transformation);

    transform.create(3, 4, CV_64F);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x)
            transform.at<double>(y, x) = transformation(y, x);
}


}
