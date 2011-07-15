#include <iostream>
#include <fstream>

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
        ("ransac_threshold,R", po::value<double>()->default_value(0.05), "RANSAC outlier rejection threshold")
        ("max_ransac_iterations,I", po::value<int>()->default_value(1000), "maximum number of RANSAC iterations")
        ("transform_file,T", po::value<std::string>(), "filename for transformation output")
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
    double ransac_threshold = vm["ransac_threshold"].as<double>();
    int max_ransac_iterations = vm["max_ransac_iterations"].as<int>();

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
    alignment.setMaximumIterations(max_ransac_iterations);
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

    Eigen::Matrix4f transformation = alignment.getFinalTransformation();
    std::cout << "Transformation: " << std::endl;
    std::cout << transformation << std::endl;

    Eigen::Matrix4f final_transformation = transformation;
    if (vm.count("transform_file"))
    {
        std::string filename = vm["transform_file"].as<std::string>();
        std::ofstream out(filename.c_str());
        if (!out.is_open()) 
        {
            std::cerr << "cannot open " << filename << " for writing." << std::endl;
            return -2;
        }
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 4; ++c)
                out << final_transformation(r, c) << " ";
            out << std::endl;
        }
        out.close();
    }

    return 0;
}
