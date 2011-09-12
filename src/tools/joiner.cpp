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
        ("source_points,P", po::value<std::string>()->required(), "PCD file for source points")
        ("source_features,F", po::value<std::string>()->required(), "PCD file for source features")
        ("target_points,Q", po::value<std::string>()->required(), "PCD file for target points")
        ("target_features,G", po::value<std::string>()->required(), "PCD file for target features")
        ("output_points,R", po::value<std::string>()->required(), "PCD file for joining result (points)")
        ("output_features,H", po::value<std::string>()->required(), "PCD file for joining result (features)")
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

    std::string source_points_file = vm["source_points"].as<std::string>();
    std::string source_features_file = vm["source_features"].as<std::string>();
    std::string target_points_file = vm["target_points"].as<std::string>();
    std::string target_features_file = vm["target_features"].as<std::string>();
    std::string output_points_file = vm["output_points"].as<std::string>();
    std::string output_features_file = vm["output_features"].as<std::string>();
    bool verbose = vm.count("verbose") > 0;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr 
        source_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(source_points_file, *source_point_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << source_points_file << "'" << std::endl;
        return -1;
    }
    if (verbose) std::cout << "Loaded " << source_point_cloud->points.size()
        << " source points from " << source_points_file << "." << std::endl;

    
    pcl::PointCloud<Descriptor>::Ptr 
        source_feature_cloud(new pcl::PointCloud<Descriptor>());
    
    if (pcl::io::loadPCDFile<Descriptor>(source_features_file, *source_feature_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << source_features_file << "'" << std::endl;
        return -1;
    }
    std::cerr << "Loaded " << source_feature_cloud->points.size()
        << " source features from " << source_features_file << "." << std::endl;
     
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr 
        target_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(target_points_file, *target_point_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << target_points_file << "'" << std::endl;
        return -1;
    }
    std::cerr << "Loaded " << target_point_cloud->points.size()
        << " target points from " << target_points_file << "." << std::endl;
    
    pcl::PointCloud<Descriptor>::Ptr 
        target_feature_cloud(new pcl::PointCloud<Descriptor>());
    
    if (pcl::io::loadPCDFile<Descriptor>(target_features_file, *target_feature_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << target_features_file << "'" << std::endl;
        return -1;
    }
    std::cerr << "Loaded " << target_feature_cloud->points.size()
        << " target features from " << target_features_file << "." << std::endl;

    pcl::KdTreeFLANN<pcl::PointXYZRGB> kd_tree;
    kd_tree.setInputCloud(target_point_cloud);

    pcl::KdTreeFLANN<Descriptor> feature_tree;
    feature_tree.setInputCloud(target_feature_cloud);
    
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr 
        output_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<Descriptor>::Ptr 
        output_feature_cloud(new pcl::PointCloud<Descriptor>());

    float join_distance = 0.05 * 0.05;
    std::vector<int> target_hit(target_point_cloud->points.size(), 0);
    for (size_t i = 0; i < source_feature_cloud->points.size(); ++i)
    {
        // find nearest feature
        int k = 2;
        std::vector<int> nn_indices(k);
        std::vector<float> nn_distances(k);
        feature_tree.nearestKSearch(source_feature_cloud->points[i], k, nn_indices, nn_distances);
        bool is_match = (nn_distances[0] / nn_distances[1]) < (0.8 * 0.8);

        target_hit[nn_indices[0]] = 1;

        float distance = euclideanDistance(source_point_cloud->points[i], 
                target_point_cloud->points[nn_indices[0]]);
        if (distance < join_distance)
        {
            output_point_cloud->points.push_back(source_point_cloud->points[i]);
            output_feature_cloud->points.push_back(source_feature_cloud->points[i]);
        }
    }



    /*
    bool binary_mode = false;
    pcl::io::savePCDFile<pcl::PointXYZRGB>(output_points_file, *output_point_cloud, binary_mode);
    std::cout << "Saved " << output_point_cloud->points.size() << " points to " << output_points_file << "." << std::endl;
    pcl::io::savePCDFile<Descriptor>(output_features_file, *output_feature_cloud);
    std::cout << "Saved " << output_feature_cloud->points.size() << " features to " << output_features_file << "." << std::endl;
    */
 
    return 0;
}

