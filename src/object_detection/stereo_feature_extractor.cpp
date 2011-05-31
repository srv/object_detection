#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"

namespace object_detection
{

StereoFeatureExtractor::StereoFeatureExtractor()
{
    feature_extractor_ = FeatureExtractorFactory::create("SURF");
    descriptor_matcher_ = cv::DescriptorMatcher::create("BruteForce");
}

std::vector<StereoFeature>
StereoFeatureExtractor::extract(
        const cv::Mat& image_left, const cv::Mat& image_right,
        double max_y_diff, double max_angle_diff, int max_size_diff)
{
    std::vector<Feature> features_left = feature_extractor_->extract(image_left);
    std::vector<Feature> features_right = feature_extractor_->extract(image_right);

    cv::Mat match_mask;
    computeMatchMask(features_left, features_right, match_mask, max_y_diff,
            max_angle_diff, max_size_diff);

    std::vector<cv::DMatch> matches;
    crossCheckMatching(features_left, features_right, matches, match_mask);
    cv::imshow("match mask", match_mask);

    std::vector<StereoFeature> stereo_features(matches.size());
    for (size_t i = 0; i < matches.size(); ++i)
    {
        int index_left = matches[i].queryIdx;
        int index_right = matches[i].trainIdx;
        stereo_features[i].feature_left = features_left[index_left];
        stereo_features[i].feature_right = features_right[index_right];
    }
    return stereo_features;
}


void StereoFeatureExtractor::computeMatchMask(
        const std::vector<Feature>& features_left,
        const std::vector<Feature>& features_right,
        cv::Mat& match_mask, double max_y_diff, double max_angle_diff, 
        int max_size_diff)
{
    if (features_left.empty() || features_right.empty())
    {
        return;
    }

    match_mask.create(features_right.size(), features_left.size(), CV_8UC1);
    for (int r = 0; r < match_mask.rows; ++r)
    {
        for (int c = 0; c < match_mask.cols; ++c)
        {
            const cv::KeyPoint& keypoint1 = features_left[c].key_point;
            const cv::KeyPoint& keypoint2 = features_right[r].key_point;
            double y_diff = fabs(keypoint1.pt.y - keypoint2.pt.y);
            double angle_diff = std::abs(keypoint1.angle - keypoint2.angle);
            angle_diff = std::min(360 - angle_diff, angle_diff);
            int size_diff = std::abs(keypoint1.size - keypoint2.size);
            if (y_diff <= max_y_diff && angle_diff <= max_angle_diff && 
                keypoint1.octave == keypoint2.octave && size_diff <= max_size_diff)
            {
                match_mask.at<unsigned char>(r, c) = 255;
            }
            else
            {
                match_mask.at<unsigned char>(r, c) = 0;
            }
        }
    }
}

void StereoFeatureExtractor::crossCheckMatching(
                            const std::vector<Feature>& features_left, 
                            const std::vector<Feature>& features_right,
                            std::vector<cv::DMatch>& matches, 
                            const cv::Mat& match_mask)
{
    if (features_left.size() == 0 || features_right.size() == 0)
        return;

    // copy descriptor data to opencv structure
    cv::Mat descriptors_left(features_left.size(), features_left[0].descriptor.size(), CV_32F);
    cv::Mat descriptors_right(features_right.size(), features_right[0].descriptor.size(), CV_32F);

    for (size_t f = 0; f < features_left.size(); ++f)
    {
        for (size_t d = 0; d < features_left[f].descriptor.size(); ++d)
        {
            descriptors_left.at<float>(f, d) = features_left[f].descriptor[d];
        }
    }

    for (size_t f = 0; f < features_right.size(); ++f)
    {
        for (size_t d = 0; d < features_right[f].descriptor.size(); ++d)
        {
            descriptors_right.at<float>(f, d) = features_right[f].descriptor[d];
        }
    }

    matches.clear();
    int knn = 3;
    std::vector<std::vector<cv::DMatch> > matches_left2right, matches_right2left;
    descriptor_matcher_->knnMatch(descriptors_left, descriptors_right,
            matches_left2right, knn, match_mask.t());
    descriptor_matcher_->knnMatch(descriptors_right, descriptors_left,
            matches_right2left, knn, match_mask);
    for (size_t m = 0; m < matches_left2right.size(); m++ )
    {
        bool cross_check_found = false;
        for (size_t fk = 0; fk < matches_left2right[m].size(); fk++ )
        {
            const cv::DMatch& forward = matches_left2right[m][fk];

            for( size_t bk = 0; bk < matches_right2left[forward.trainIdx].size(); bk++ )
            {
                const cv::DMatch& backward = matches_right2left[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    matches.push_back(forward);
                    cross_check_found = true;
                    break;
                }
            }
            if (cross_check_found) break;
        }
    }
}

void StereoFeatureExtractor::setFeatureExtractor(
        FeatureExtractor::Ptr& feature_extractor)
{
    feature_extractor_ = feature_extractor;
}


const int draw_shift_bits = 4;
const int draw_multiplier = 1 << draw_shift_bits;
/*
 * Functions to draw keypoints and matches.
 */
static inline void _drawKeypoint(cv::Mat& img, const cv::KeyPoint& p, const cv::Scalar& color, int flags )
{
    cv::Point center(cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier) );

    if( flags & cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS )
    {
        int radius = cvRound(p.size/2 * draw_multiplier); // KeyPoint::size is a diameter

        // draw the circles around keypoints with the keypoints size
        cv::circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );

        // draw orientation of the keypoint, if it is applicable
        if( p.angle != -1 )
        {
            float srcAngleRad = p.angle*(float)CV_PI/180.f;
            cv::Point orient(cvRound(cos(srcAngleRad)*radius), 
						 cvRound(sin(srcAngleRad)*radius));
            cv::line( img, center, center+orient, color, 1, CV_AA, draw_shift_bits );
        }
#if 0
        else
        {
            // draw center with R=1
            int radius = 1 * draw_multiplier;
            circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
        }
#endif
    }
    else
    {
        // draw center with R=3
        int radius = 3 * draw_multiplier;
        cv::circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
    }
}

void paintStereoFeatureMatchings(cv::Mat& image, const cv::Mat& image_left, 
        const cv::Mat& image_right, 
        const std::vector<StereoFeature>& stereo_features)
{
    assert(image_left.type() == CV_8UC3);
    assert(image_right.type() == CV_8UC3);

    // y shift (so that we see diagonal instead of straight lines
    int y_shift = image_right.rows / 5;

    image.create(image_left.rows + y_shift, image_left.cols + image_right.cols, CV_8UC3);
    image = cv::Scalar::all(0);
    
    // copy images
    cv::Mat image_left_hdr = image(cv::Rect(0, 0, image_left.cols, image_left.rows));
    cv::Mat image_right_hdr = image(cv::Rect(image_left.cols, y_shift, image_right.cols, image_right.rows));
    image_left.copyTo(image_left_hdr);
    image_right.copyTo(image_right_hdr);

    cv::RNG& rng = cv::theRNG();
    
    for (size_t i = 0; i < stereo_features.size(); ++i)
    {
        cv::Point2d p1 = stereo_features[i].feature_left.key_point.pt;
        cv::Point2d p2 = stereo_features[i].feature_right.key_point.pt;
        p2.x += image_left.cols;
        p2.y += y_shift;
        int thickness = 1;
        cv::Scalar color = cv::Scalar(rng(256), rng(256), rng(256));
        cv::line(image, p1, p2, color, thickness);
        _drawKeypoint(image_left_hdr, stereo_features[i].feature_left.key_point, color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        _drawKeypoint(image_right_hdr, stereo_features[i].feature_right.key_point, color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
}


} // namespace object_detection

