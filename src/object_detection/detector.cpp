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

    is_trained_ = true;
}

std::vector<Detection> Detector::detect(const cv::Mat& image,
        const std::vector<cv::Rect>& rois)
{
    if (!is_trained_)
    {
        throw std::runtime_error("Error: Detector::detect() called without having trained before!");
    }

    std::vector<Detection> all_detections;

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
    return all_detections;
}


