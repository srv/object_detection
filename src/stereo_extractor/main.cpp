#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>

#include "feature_extractor_factory.h"
#include "stereo_feature_extractor.h"

using namespace object_detection;
namespace po = boost::program_options;

int main(int argc, char **argv){

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("left,l", po::value<string>()->required(), "left image")
        ("right,r", po::value<string>()->required(), "right image")
        ("max_y_diff,y", po::value<double>()->default_value(2.0), "maximum y difference for matching keypoints")
        ("max_angle_diff,a", po::value<double>()->default_value(4.0), "maximum angle difference for matching keypoints")
        ("max_size_diff,s", po::value<int>()->default_value(5), "maximum size difference for matching keypoints")
        ("feature_extractor,f", po::value<string>()->default_value("SURF"), "feature extractor")
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


    std::string left_image_file = vm["left"].as<std::string>();
    std::string right_image_file = vm["right"].as<std::string>();
    double max_y_diff = vm["max_y_diff"].as<double>();
    double max_angle_diff = vm["max_angle_diff"].as<double>();
    int max_size_diff = vm["max_size_diff"].as<int>();
    std::string feature_extractor_name = vm["feature_extractor"].as<std::string>();


    FeatureExtractor::Ptr feature_extractor = 
        FeatureExtractorFactory::create(feature_extractor_name);
    if (feature_extractor.get() == 0)
    {
        std::cerr << "Cannot create feature extractor with name '" 
            << feature_extractor_name << "'" << std::endl;
        return -2;
    }

    cv::Mat image_left = cv::imread(left_image_file);
    cv::Mat image_right = cv::imread(right_image_file);

    StereoFeatureExtractor extractor;
    extractor.setFeatureExtractor(feature_extractor);
    std::cout << "Running extractor..." << std::flush;
    std::vector<StereoFeature> stereo_features = 
        extractor.extract(image_left, image_right, max_y_diff, max_angle_diff, max_size_diff);
    std::cout << "found " << stereo_features.size() << " stereo features." << std::endl;

    cv::Mat result_image;
    paintStereoFeatureMatchings(result_image, image_left, image_right, stereo_features);

    cv::namedWindow("matchings", CV_WINDOW_NORMAL);
    cv::imshow("matchings", result_image);
    cvWaitKey();


    return 0;
}

