#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

namespace po = boost::program_options;

bool loadTransformation(const std::string& filename, Eigen::Matrix4f& transformation)
{
    std::ifstream in(filename.c_str());
    if (!in.is_open()) return false;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
        {
            double value;
            in >> value;
            transformation(r, c) = value;
        }
    return true;
}

double computeTranslationError(const Eigen::Matrix4f& true_transform, const Eigen::Matrix4f& estimated_transform)
{
    Eigen::Vector4f real_trans = true_transform.col(3);
    Eigen::Vector4f est_trans = estimated_transform.col(3);
    Eigen::Vector4f diff = real_trans - est_trans;
    return diff.norm();
}

double computeRotationAxisError(const Eigen::Matrix4f& true_transform, const Eigen::Matrix4f& estimated_transform, bool verbose)
{
    Eigen::AngleAxisf aa_real;
    aa_real.fromRotationMatrix(true_transform.topLeftCorner<3, 3>());
    if (verbose) std::cout << "Real Axis: " << aa_real.axis() << std::endl;
    Eigen::AngleAxisf aa_est;
    aa_est.fromRotationMatrix(estimated_transform.topLeftCorner<3, 3>());
    if (verbose) std::cout << "Est. Axis: " << aa_est.axis() << std::endl;
    return 1.0 - aa_est.axis().dot(aa_real.axis());
}

double computeRotationAngleError(const Eigen::Matrix4f& true_transform, const Eigen::Matrix4f& estimated_transform, bool verbose)
{
    Eigen::AngleAxisf aa_real;
    aa_real.fromRotationMatrix(true_transform.topLeftCorner<3, 3>());
    if (verbose) std::cout << "Real Angle: " << aa_real.angle() << std::endl;
    Eigen::AngleAxisf aa_est;
    aa_est.fromRotationMatrix(estimated_transform.topLeftCorner<3, 3>());
    if (verbose) std::cout << "Est. Angle: " << aa_est.angle() << std::endl;
    return std::abs(aa_est.angle() - aa_real.angle());
}

int main(int argc, char** argv)
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("true_transformation,T", po::value<std::string>()->required(), "transformation file that defines the ground truth")
        ("estimated_transformation,E", po::value<std::string>()->required(), "transformation file that defines the estimation")
        ("points,P", po::value<std::string>(), "PCD file for points (used for reprojection error if given)")
        ("verbose,V", "verbose output")
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

    std::string true_transformation_file = vm["true_transformation"].as<std::string>();
    std::string estimated_transformation_file = vm["estimated_transformation"].as<std::string>();
    bool verbose = false;
    if (vm.count("verbose")) verbose = true;

    Eigen::Matrix4f true_transformation;
    if (!loadTransformation(true_transformation_file, true_transformation))
    {
        std::cerr << "Cannot load transformation from " << true_transformation_file << "." << std::endl;
        return -2;
    }
    if (verbose) std::cout << "Loaded ground truth transformation:\n" << true_transformation << std::endl;
    Eigen::Matrix4f estimated_transformation;
    if (!loadTransformation(estimated_transformation_file, estimated_transformation))
    {
        std::cerr << "Cannot load transformation from " << true_transformation_file << "." << std::endl;
        return -2;
    }
    if (verbose) std::cout << "Loaded estimated transformation:\n" << estimated_transformation << std::endl;

    std::cout << "Translation error: " << computeTranslationError(true_transformation, estimated_transformation) << std::endl;
    std::cout << "Rotation axis error (0 = same axis, 1 = perpendicular axes): "
        << computeRotationAxisError(true_transformation, estimated_transformation, verbose) << std::endl;
    std::cout << "Rotation angle error (difference in 'amount' of rotation): "
        << computeRotationAngleError(true_transformation, estimated_transformation, verbose) << std::endl;

    if (vm.count("points"))
    {
        std::string points_file = vm["points"].as<std::string>();
        pcl::PointCloud<pcl::PointXYZ>::Ptr 
            point_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(points_file, *point_cloud) == -1)
        {
            std::cerr << "Couldn't read file '" << points_file << "'" << std::endl;
            return -1;
        }
        if (verbose)
        {
            std::cout << "Loaded " << point_cloud->points.size()
            << " points from " << points_file << "." << std::endl;
        }
        double error = 0.0;
        for (size_t i = 0; i < point_cloud->points.size(); ++i)
        {
            pcl::PointXYZ& point = point_cloud->points[i];
            Eigen::Vector4f vec(point.x, point.y, point.z, 1.0);
            Eigen::Vector4f ideal_point = true_transformation * vec;
            Eigen::Vector4f estimated_point = estimated_transformation * vec;
            Eigen::Vector4f dist = ideal_point - estimated_point;
            error += dist.norm();
        }
        error /= point_cloud->points.size();
        std::cout << "Mean reprojection error (distance from true to estimated point): " << error << std::endl;

    }
    return 0;
}
