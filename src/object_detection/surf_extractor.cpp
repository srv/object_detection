#include <opencv2/imgproc/imgproc.hpp>

#include "surf_extractor.h"

using object_detection::SurfExtractor;
using object_detection::Feature;


SurfExtractor::SurfExtractor()
{
}


std::vector<Feature>
SurfExtractor::extract(const cv::Mat& image)
{
    assert(image.type() == CV_8UC3 || image.type() == CV_8U);

    cv::Mat gray_image;
    if (image.type() == CV_8UC3)
    {
        cv::cvtColor(image, gray_image, CV_BGR2GRAY);
    }
    else
    {
        gray_image = image;
    }

    std::vector<float> descriptor_data;
    std::vector<cv::KeyPoint> key_points;
    surf_(gray_image, cv::Mat(), key_points, descriptor_data);

    std::vector<Feature> features(key_points.size());

    for (size_t i = 0; i < key_points.size(); ++i)
    {
        features[i].key_point = key_points[i];
        features[i].descriptor.resize(surf_.descriptorSize());
        std::copy(&descriptor_data[i], 
                &descriptor_data[i] + surf_.descriptorSize(), 
                features[i].descriptor.begin());
    }
    
    return features;
}
