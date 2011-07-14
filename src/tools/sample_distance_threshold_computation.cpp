#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/feature.h> // for compute3DCentroid

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

float 
computeSampleDistanceThreshold (const PointCloud &cloud)
{
    // Compute the principal directions via PCA
    Eigen::Vector4f xyz_centroid;
    pcl::compute3DCentroid (cloud, xyz_centroid);
    EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
    computeCovarianceMatrixNormalized (cloud, xyz_centroid, covariance_matrix);
    EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
    EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
    pcl::eigen33 (covariance_matrix, eigen_vectors, eigen_values);

    // Compute the distance threshold for sample selection
    //sample_dist_thresh_ = eigen_values.array ().sqrt ().sum () / 3.0;
    //sample_dist_thresh_ *= sample_dist_thresh_;
    return eigen_values.array ().sqrt ().sum () / 3.0;
}

float
computeSampleDistanceThreshold2 (const PointCloud &cloud)
{
    pcl::PointXYZ centroid(0, 0, 0);
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        centroid.x += cloud.points[i].x;
        centroid.y += cloud.points[i].y;
        centroid.z += cloud.points[i].z;
    }
    centroid.x /= cloud.points.size();
    centroid.y /= cloud.points.size();
    centroid.z /= cloud.points.size();

    float distances = 0.0;
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        distances += pcl::euclideanDistance(cloud.points[i], centroid);
    }
    return distances / cloud.points.size();
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <pcd file> " << std::endl;
        return -1;
    }

    std::string filename(argv[1]);

    PointCloud point_cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, point_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << filename << "'" << std::endl;
        return -1;
    }
    std::cout << "Loaded " << point_cloud.points.size()
        << " points from " << filename << "." << std::endl;

    float thresh1 = computeSampleDistanceThreshold(point_cloud);
    float thresh2 = computeSampleDistanceThreshold2(point_cloud);
    std::cout << "thresh1 = " << thresh1 << std::endl;
    std::cout << "thresh2 = " << thresh2 << std::endl;

    return 0;
}

