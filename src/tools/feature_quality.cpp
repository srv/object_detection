#include <iostream>

#include <boost/program_options.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/feature.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>


#include "descriptor.h"

namespace po = boost::program_options;
using object_detection::Descriptor;

int main(int argc, char** argv)
{
    srand(time(0));
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("features,F", po::value<std::string>()->required(), "PCD file for features")
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

    std::string features_file = vm["features"].as<std::string>();

    pcl::PointCloud<Descriptor>::Ptr 
        feature_cloud(new pcl::PointCloud<Descriptor>());
    
    if (pcl::io::loadPCDFile<Descriptor>(features_file, *feature_cloud) == -1)
    {
        std::cerr << "Couldn't read file '" << features_file << "'" << std::endl;
        return -1;
    }
    std::cerr << "Loaded " << feature_cloud->points.size()
        << " features from " << features_file << "." << std::endl;
     
    pcl::KdTreeFLANN<Descriptor> kd_tree;
    kd_tree.setInputCloud(feature_cloud);

    int k = 2;
    std::vector<int> nn_indices(k);
    std::vector<float> nn_distances(k);
    std::vector<float> distances(feature_cloud->points.size());
    for (size_t i = 0; i < feature_cloud->points.size(); ++i)
    {
        kd_tree.nearestKSearch(i, k, nn_indices, nn_distances);
        assert(nn_indices[0] == (int)i); // to be sure...
        assert(nn_distances[0] == 0);
        distances[i] = nn_distances[1];
    }
    std::sort(distances.begin(), distances.end());
    for (size_t i = 0; i < distances.size(); ++i)
        std::cout << 100.0f * i / distances.size() << "% below " << distances[i] << std::endl;

    return 0;
}
