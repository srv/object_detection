
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>
#include <pcl/features/feature.h>

#include "feature.h"
#include "model.h"

#include "model_builder.h"

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

void ModelBuilder::extend(Model& model, 
        const std::vector<cv::Point3f>& points,
        const std::vector<Feature>& features)
{
    if (features.size() == 0)
    {
        return;
    }

    // create descriptor matrix
    size_t descriptor_size = features[0].descriptor.size();
    cv::Mat new_features_mat(features.size(), 
            descriptor_size, CV_32F);
    for (size_t i = 0; i < features.size(); ++i)
    {
         std::copy(features[i].descriptor.begin(),
                   features[i].descriptor.end(),
                   new_features_mat.ptr<float>(i));
    }

    // match feature descriptors
    cv::Mat model_features_mat = model.getFeatureData();
    std::vector<cv::DMatch> matches;
    matchFeatures(new_features_mat, model_features_mat, matches);

    // grab corresponding points
    std::vector<cv::Point3f> matched_model_points;
    std::vector<cv::Point3f> matched_new_points;
    std::vector<int> matched_point_indices;
    for (size_t i = 0; i < matches.size(); ++i)
    {
        matched_model_points.push_back(
                model.getWorldPoint(matches[i].trainIdx));
        matched_new_points.push_back(points[matches[i].queryIdx]);
        matched_point_indices.push_back(matches[i].queryIdx);
    }
    std::sort(matched_point_indices.begin(),
            matched_point_indices.end());

    // calculate transformation for matching points
    cv::Mat transform;
    calculateTransform(matched_new_points, matched_model_points,
            transform);

    // apply transformation
    std::vector<cv::Point3f> new_model_points;
    // transform(points, new_model_points, transform)

    // extend model
    for (size_t i = 0; i < new_model_points.size(); ++i)
    {
        // look if point was not matched
        if (!std::binary_search(matched_point_indices.begin(),
                    matched_point_indices.end(), i))
        {
            // point is new -> add to model
            model.addFeature(new_model_points[i], features[i]);
        }
    }
}

void ModelBuilder::matchFeatures(const cv::Mat& features1, 
        const cv::Mat& features2, std::vector<cv::DMatch>& matches)
{
    assert(features1.cols == features2.cols);

    matches.clear();

    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = 
        cv::DescriptorMatcher::create("BruteForce");
    std::vector<std::vector<cv::DMatch> > knn_matches;
    cv::Mat query_features = features1;
    cv::Mat training_features = features2;
    descriptor_matcher->knnMatch(query_features, training_features, 
            knn_matches, 2);

    for (size_t i = 0; i < knn_matches.size(); ++i)
    {
        if (knn_matches[i].size() == 2)
        {
            const cv::DMatch& match1 = knn_matches[i][0];
            const cv::DMatch& match2 = knn_matches[i][1];
            if (match1.distance / match2.distance < 0.8)
            {
                matches.push_back(match1);
            }
        }
    }
}


void ModelBuilder::calculateTransform(
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
