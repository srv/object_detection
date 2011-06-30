#include <fstream>
#include <stdexcept>

#include <highgui.h>

#include <boost/program_options.hpp>

#include "detector.h"
#include "detection.h"
#include "object_parts_detector.h"
#include "colored_parts_classifier.h"
#include "textured_parts_classifier.h"
#include "training_data.h"
#include "utilities.h"
#include "statistics.h"

using object_detection::Detector;
using object_detection::Detection;
using object_detection::paintFilledPolygon;

namespace po = boost::program_options;

Detector::Detector(const std::string& config_file_name) : 
    is_trained_(false),
    config_file_name_(config_file_name)
{
    setup();
}

void Detector::setup()
{
    // read config file
    std::ifstream config_file_stream(config_file_name_.c_str());
    if (!config_file_stream.is_open())
    {
        std::cerr << "Detector::setup(): ERROR: given config file "
                << config_file_name_ << " could not be opened." << std::endl;
    }
    po::options_description config_file_options;
    config_file_options.add_options()
        ("num_hue_bins", po::value<int>(), "set number of hue bins")
        ("num_saturation_bins", po::value<int>(), "set number of saturation bins")
        ("min_saturation", po::value<int>(), "set minimum saturation")
        ("min_value", po::value<int>(), "set minimum value")
        ("min_color_occurences", po::value<int>(), "set minimum occurences of colors")
        ;
    po::variables_map variables_map;
    po::store(po::parse_config_file(config_file_stream, config_file_options), variables_map);
    po::notify(variables_map);
    
    boost::shared_ptr<ColoredPartsClassifier> colored_parts_classifier = 
        boost::shared_ptr<ColoredPartsClassifier>(new ColoredPartsClassifier());

    if (variables_map.count("num_hue_bins"))
    {
        colored_parts_classifier->setNumHueBins(
                variables_map["num_hue_bins"].as<int>());
    }
    if (variables_map.count("num_saturation_bins"))
    {
        colored_parts_classifier->setNumSaturationBins(
                variables_map["num_saturation_bins"].as<int>());
    }
    if (variables_map.count("min_saturation"))
    {
        colored_parts_classifier->setMinSaturation(
                variables_map["min_saturation"].as<int>());
    }
    if (variables_map.count("min_value"))
    {
        colored_parts_classifier->setMinValue(
                variables_map["min_value"].as<int>());
    }
    if (variables_map.count("min_color_occurences"))
    {
        colored_parts_classifier->setMinOccurences(
                variables_map["min_color_occurences"].as<int>());
    }

    std::cout << "Detector::setup(): num_hue_bins set to " <<
                                     colored_parts_classifier->numHueBins() << std::endl;
    std::cout << "Detector::setup(): num_saturation_bins set to " <<
                                     colored_parts_classifier->numSaturationBins() << std::endl;
    std::cout << "Detector::setup(): min_saturation set to " <<
                                     colored_parts_classifier->minSaturation() << std::endl;
    std::cout << "Detector::setup(): min_value set to " <<
                                     colored_parts_classifier->minValue() << std::endl;
    std::cout << "Detector::setup(): min_color_occurences set to " <<
                                     colored_parts_classifier->minOccurences() << std::endl;


    boost::shared_ptr<ObjectPartsDetector> color_parts_detector =
        boost::shared_ptr<ObjectPartsDetector>(new ObjectPartsDetector(colored_parts_classifier));

    object_parts_detectors_.push_back(color_parts_detector);

    /*

    boost::shared_ptr<TexturedPartsClassifier> textured_parts_classifier = 
        boost::shared_ptr<TexturedPartsClassifier>(new TexturedPartsClassifier());
    
    boost::shared_ptr<ObjectPartsDetector> textured_parts_detector =
        boost::shared_ptr<ObjectPartsDetector>(new ObjectPartsDetector(textured_parts_classifier));


    object_parts_detectors_.push_back(textured_parts_detector);

    */
}

void Detector::train(const TrainingData& training_data)
{
    // check input
    if (!training_data.isValid())
    {
        throw std::runtime_error("Detector::train(): input data invalid");
    }

    // create object mask
    cv::Mat object_mask = cv::Mat::zeros(training_data.image.rows, 
            training_data.image.cols, CV_8UC1);
    paintFilledPolygon(object_mask, training_data.object_outline,
            cv::Scalar(255));

    for(size_t i = 0; i < object_parts_detectors_.size(); ++i)
    {
        object_parts_detectors_[i]->train(training_data.image, object_mask);
    }

    cv::Mat outline_points_matrix = cv::Mat(training_data.object_outline);
    cv::Scalar centroid = cv::mean(outline_points_matrix);
     
    cv::Mat centered_outline_points_matrix = outline_points_matrix - centroid;
    centered_object_outline_ = cv::Mat_<cv::Point>(centered_outline_points_matrix);

    // 3d information
    for (size_t i = 0; i < training_data.stereo_features.size(); ++i)
    {
        const StereoFeature& stereo_feature = training_data.stereo_features[i];
        Feature feature;
        feature.key_point = stereo_feature.key_point;
        feature.descriptor = stereo_feature.descriptor;
        object_model_.addFeature(stereo_feature.world_point, feature);
    }

    is_trained_ = true;
}

std::vector<Detection> Detector::detect(const cv::Mat& image,
        const std::vector<StereoFeature>& stereo_features,
        const std::vector<cv::Rect>& rois)
{
    std::cout << "Detector::detect: " << image.rows << "x" << image.cols 
        << " image, " << stereo_features.size() << " features, "
        << rois.size() << " rois." << std::endl;

    if (!is_trained_)
    {
        throw std::runtime_error("Error: Detector::detect() called without having trained before!");
    }

    std::vector<Detection> all_detections;

    /*
    for(size_t i = 0; i < object_parts_detectors_.size(); ++i)
    {
        std::vector<Detection> detections = object_parts_detectors_[i]->detect(image);

        // compute outline for detections as parts detector do not know
        // about outlines
        // TODO change this!?
        for (size_t i = 0; i < detections.size(); ++i)
        {
            Detection& detection = detections[i]; 
            cv::Mat rotation_matrix = cv::getRotationMatrix2D(cv::Point2f(0.0, 0.0),
                    -detection.angle / M_PI * 180.0, detection.scale);
            cv::Mat rotated_points_matrix;
            cv::transform(cv::Mat(centered_object_outline_), rotated_points_matrix, 
                    rotation_matrix);
            cv::add(rotated_points_matrix, 
                    cv::Scalar(detection.center.x, detection.center.y), 
                    rotated_points_matrix);
            detection.outline = cv::Mat_<cv::Point>(rotated_points_matrix);
            cv::Scalar new_center = cv::mean(cv::Mat(detection.outline));
            detection.center.x = new_center[0];
            detection.center.y = new_center[1];
        }
        all_detections.insert(all_detections.end(), detections.begin(), detections.end());
    }
    */

    /*

    if (stereo_features.size() > 0)
    {
        // 3d information
        Model scene;
        for (size_t i = 0; i < stereo_features.size(); ++i)
        {
            Feature feature;
            feature.key_point = stereo_features[i].key_point;
            feature.descriptor = stereo_features[i].descriptor;
            scene.addFeature(stereo_features[i].world_point, feature);
        }
    }
    */
    return all_detections;
}

bool Detector::estimatePose(const Model& object_model, const Model& scene_model, cv::Mat& transformation)
{
    // 3d matching
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = cv::DescriptorMatcher::create("BruteForce");
    std::vector<std::vector<cv::DMatch> > matches;
    cv::Mat training_features = scene_model.getFeatureData();
    cv::Mat query_features = object_model.getFeatureData();
    descriptor_matcher->knnMatch(query_features, training_features, matches, 2);

    std::vector<cv::Point3f> matched_object_points;
    std::vector<cv::Point3f> matched_scene_points;

    for (size_t i = 0; i < matches.size(); ++i)
    {
        if (matches[i].size() == 2)
        {
            const cv::DMatch& match1 = matches[i][0];
            const cv::DMatch& match2 = matches[i][1];
            if (match1.distance / match2.distance < 0.8)
            {
                matched_scene_points.push_back(scene_model.getWorldPoint(match1.trainIdx));
                matched_object_points.push_back(object_model.getWorldPoint(match1.queryIdx));
            }
        }
    }
    std::cout << matched_object_points.size() << " matching object/world points." << std::endl;

    if (matched_object_points.size() > 3)
    {
        std::vector<uchar> outliers;
        double ransac_threshold = 300.0;
        int ret = cv::estimateAffine3D(cv::Mat(matched_object_points), cv::Mat(matched_scene_points), 
                transformation, outliers, ransac_threshold, 0.5);

        std::cout << "..............." << ret << "...................." << std::endl;
        //std::cout << "Transformation: " << transformation << std::endl;
        int num_outliers = 0;
        int num_inliers = 0;
        for (size_t i = 0; i < outliers.size(); ++i)
        {
            if (outliers[i] > 0) num_outliers++; else num_inliers++;
        }
        std::cout << std::endl << num_outliers << " outliers, " << num_inliers << " inliers." << std::endl;
        return true;
    }
    else
    {
        return false;
    }
}

