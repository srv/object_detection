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
        ("num_samples,N", po::value<int>()->default_value(3), "number of samples for RANSAC")
        ("min_sample_distance,D", po::value<double>()->default_value(0.1), "minimum distance of samples")
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
    int num_samples = vm["num_samples"].as<int>();
    double min_sample_distance = vm["min_sample_distance"].as<double>();
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

    pcl::MySampleConsensusInitialAlignment<pcl::PointXYZRGB, pcl::PointXYZRGB, Descriptor> sac_ia;
    sac_ia.setNumberOfSamples(num_samples);
    sac_ia.setMaximumIterations(max_alignment_iterations);
    sac_ia.setMinSampleDistance(min_sample_distance);
    sac_ia.setRANSACOutlierRejectionThreshold(ransac_threshold);
    std::cout << "number of samples = " << sac_ia.getNumberOfSamples() << std::endl;
    std::cout << "minimum sample distance = " << sac_ia.getMinSampleDistance() << std::endl;
    std::cout << "max iterations = " << sac_ia.getMaximumIterations() << std::endl;
    std::cout << "ransac outlier threshold = " << sac_ia.getRANSACOutlierRejectionThreshold() << std::endl;
    std::cout << "max correspondence distance = " << sac_ia.getMaxCorrespondenceDistance() << std::endl;
    std::cout << "transformation epsilon = " << sac_ia.getTransformationEpsilon() << std::endl;
    sac_ia.setInputCloud(model_point_cloud);
    sac_ia.setSourceFeatures(model_feature_cloud);
    sac_ia.setInputTarget(scene_point_cloud);
    sac_ia.setTargetFeatures(scene_feature_cloud);
    sac_ia.setMaxCorrespondenceDistance(0.5);
    sac_ia.setMinSampleDistance(0.0);
    // set point representation of descriptor to use all 64 values (default is 3)
    //pcl::CustomPointRepresentation<Descriptor>::Ptr descriptor_point_representation(64);
    //sac_ia.setPointRepresentation(descriptor_point_representation.makeShared());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr alignment_output(new pcl::PointCloud<pcl::PointXYZRGB>());
    sac_ia.align(*alignment_output);
    std::cout << "has converged = " << sac_ia.hasConverged() << std::endl;

    Eigen::Matrix4f transformation = sac_ia.getFinalTransformation();
    std::cout << "Transformation: " << std::endl;
    std::cout << transformation << std::endl;
    std::cout << "fitness score = " << sac_ia.getFitnessScore() << std::endl;

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

    return 0;
}