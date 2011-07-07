#include "feature_matcher.h"

#include <opencv2/features2d/features2d.hpp>

namespace object_detection
{

void FeatureMatcher::matchFeatures(const cv::Mat& features1, 
        const cv::Mat& features2, std::vector<cv::DMatch>& matches)
{
    assert(features1.cols == features2.cols);

    matches.clear();

    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = 
        cv::DescriptorMatcher::create("BruteForce");
    std::vector<std::vector<cv::DMatch> > knn_matches;
    cv::Mat query_features = features1;
    cv::Mat training_features = features2;
    descriptor_matcher->knnMatch(query_features, training_features, 
            knn_matches, 2);

    for (size_t i = 0; i < knn_matches.size(); ++i)
    {
        if (knn_matches[i].size() == 2)
        {
            const cv::DMatch& match1 = knn_matches[i][0];
            const cv::DMatch& match2 = knn_matches[i][1];
            if (match1.distance / match2.distance < 0.8)
            {
                matches.push_back(match1);
            }
        }
    }
}

}

