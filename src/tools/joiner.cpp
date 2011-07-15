#include <iostream>

#include <boost/program_options.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/feature.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/registration/icp.h>

#include "alignment.h"
#include "descriptor.h"


namespace po = boost::program_options;
using object_detection::Descriptor;

int main(int argc, char** argv)
{
    srand(time(0));
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("scene_points,P", po::value<std::string>()->required(), "PCD file for scene points")
        ("scene_features,F", po::value<std::string>()->required(), "PCD file for scene features")
        ("model_points,Q", po::value<std::string>()->required(), "PCD file for model points")
        ("model_features,G", po::value<std::string>()->required(), "PCD file for model features")
        ("output_points,R", po::value<std::string>()->required(), "PCD file for registration result (points)")
        ("output_features,H", po::value<std::string>()->required(), "PCD file for registration result (features)")
        ("num_samples,N", po::value<int>()->default_value(3), "number of samples for RANSAC")
        ("ransac_threshold,T", po::value<double>()->default_value(0.05), "ransac outlier rejection threshold")
        ("max_alignment_iterations,I", po::value<int>()->default_value(1000), "maximum number of RANSAC iterations for initial alignment")
        ("max_icp_iterations,J", po::value<int>()->default_value(1000), "maximum number of iterations for icp")
        ("max_icp_distance,M", po::value<double>()->default_value(0.1), "maximum distance for correspondences in ICP");
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

    std::string scene_points_file = vm["scene_points"].as<std::string>();
    std::string scene_features_file = vm["scene_features"].as<std::string>();
    std::string model_points_file = vm["model_points"].as<std::string>();
    std::string model_features_file = vm["model_features"].as<std::string>();
    std::string output_points_file = vm["output_points"].as<std::string>();
    std::string output_features_file = vm["output_features"].as<std::string>();
    int num_samples = vm["num_samples"].as<int>();
    double ransac_threshold = vm["ransac_threshold"].as<double>();
    int max_alignment_iterations = vm["max_alignment_iterations"].as<int>();
    int max_icp_iterations = vm["max_icp_iterations"].as<int>();
    double max_icp_distance = vm["max_icp_distance"].as<double>();


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr 
        scene_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(scene_points_file, *scene_point_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << scene_points_file << "'" << std::endl;
        return -1;
    }
    std::cerr << "Loaded " << scene_point_cloud->points.size()
        << " scene points from " << scene_points_file << "." << std::endl;

    
    pcl::PointCloud<Descriptor>::Ptr 
        scene_feature_cloud(new pcl::PointCloud<Descriptor>());
    
    if (pcl::io::loadPCDFile<Descriptor>(scene_features_file, *scene_feature_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << scene_features_file << "'" << std::endl;
        return -1;
    }
    std::cerr << "Loaded " << scene_feature_cloud->points.size()
        << " scene features from " << scene_features_file << "." << std::endl;
     
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr 
        model_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(model_points_file, *model_point_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << model_points_file << "'" << std::endl;
        return -1;
    }
    std::cerr << "Loaded " << model_point_cloud->points.size()
        << " model points from " << model_points_file << "." << std::endl;

    
    pcl::PointCloud<Descriptor>::Ptr 
        model_feature_cloud(new pcl::PointCloud<Descriptor>());
    
    if (pcl::io::loadPCDFile<Descriptor>(model_features_file, *model_feature_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << model_features_file << "'" << std::endl;
        return -1;
    }
    std::cerr << "Loaded " << model_feature_cloud->points.size()
        << " model features from " << model_features_file << "." << std::endl;

    object_detection::Alignment<pcl::PointXYZRGB, pcl::PointXYZRGB, Descriptor> alignment;
    alignment.setNumberOfSamples(num_samples);
    alignment.setMaximumIterations(max_alignment_iterations);
    alignment.setRANSACOutlierRejectionThreshold(ransac_threshold);
    std::cout << "number of samples = " << alignment.getNumberOfSamples() << std::endl;
    std::cout << "max iterations = " << alignment.getMaximumIterations() << std::endl;
    std::cout << "ransac outlier threshold = " << alignment.getRANSACOutlierRejectionThreshold() << std::endl;
    std::cout << "max correspondence distance = " << alignment.getMaxCorrespondenceDistance() << std::endl;
    std::cout << "transformation epsilon = " << alignment.getTransformationEpsilon() << std::endl;
    alignment.setInputCloud(model_point_cloud);
    alignment.setSourceFeatures(model_feature_cloud);
    alignment.setInputTarget(scene_point_cloud);
    alignment.setTargetFeatures(scene_feature_cloud);
    alignment.setMaxCorrespondenceDistance(0.5);
    // set point representation of descriptor to use all 64 values (default is 3)
    //pcl::CustomPointRepresentation<Descriptor>::Ptr descriptor_point_representation(64);
    //alignment.setPointRepresentation(descriptor_point_representation.makeShared());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr alignment_output(new pcl::PointCloud<pcl::PointXYZRGB>());
    alignment.align(*alignment_output);
    std::cout << "has converged = " << alignment.hasConverged() << std::endl;

    Eigen::Matrix4f transformation = alignment.getFinalTransformation();
    std::cout << "Transformation: " << std::endl;
    std::cout << transformation << std::endl;
    std::cout << "fitness score = " << alignment.getFitnessScore() << std::endl;

    // run ICP
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setMaximumIterations(max_icp_iterations);
    icp.setMaxCorrespondenceDistance(max_icp_distance);
    std::cout << "ICP max iterations = " << icp.getMaximumIterations() << std::endl;
    std::cout << "ICP ransac outlier threshold = " << icp.getRANSACOutlierRejectionThreshold() << std::endl;
    std::cout << "ICP max correspondence distance = " << icp.getMaxCorrespondenceDistance() << std::endl;
    std::cout << "ICP transformation epsilon = " << icp.getTransformationEpsilon() << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB> registration_output;
    icp.setInputCloud(alignment_output);
    icp.setInputTarget(scene_point_cloud);
    icp.align(registration_output);
    std::cout << "ICP transformation: \n" << icp.getFinalTransformation() << std::endl;
    std::cout << "ICP fitness score = " << icp.getFitnessScore() << std::endl;
    std::cout << "has converged = " << icp.hasConverged() << std::endl;

    std::cout << "final transformation: \n" << transformation * icp.getFinalTransformation() << std::endl;

    // join scene with transformed model
    scene_point_cloud->width += registration_output.width;
    scene_feature_cloud->width += registration_output.width;
    for (size_t i = 0; i < registration_output.points.size(); ++i)
    {
        scene_point_cloud->points.push_back(registration_output.points[i]);
        scene_feature_cloud->points.push_back(model_feature_cloud->points[i]);
    }

    bool binary_mode = false;
    pcl::io::savePCDFile<pcl::PointXYZRGB>(output_points_file, *scene_point_cloud, binary_mode);
    std::cout << "Saved " << scene_point_cloud->points.size() << " points to " << output_points_file << "." << std::endl;
    pcl::io::savePCDFile<Descriptor>(output_features_file, *scene_feature_cloud);
    std::cout << "Saved " << scene_feature_cloud->points.size() << " features to " << output_features_file << "." << std::endl;
 
    return 0;
}
