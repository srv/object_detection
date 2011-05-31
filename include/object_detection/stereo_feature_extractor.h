#ifndef STEREO_FEATURE_EXTRACTOR_H
#define STEREO_FEATURE_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>

#include "feature.h"
#include "feature_extractor.h"

namespace object_detection
{

/**
* Struct to hold a stereo matching result
*/
struct StereoFeature
{
    Feature feature_left;
    Feature feature_right;
};


/**
* \class StereoFeatureExtractor
* \brief Extractor for matching keypoints in a rectified stereo image pair
*/
class StereoFeatureExtractor
{

  public:

    /**
    * Constructor creates default feature detector, descriptor extractor
    * and descriptor matcher
    */
    StereoFeatureExtractor();

    /**
    * Extracts stereo keypoints from given rectified stereo image pair.
    * Keypoints for each image are computed, a match mask that preserves
    * the epipolar constraints (given by max* parameters) is computed, 
    * descriptors for keypoints are computed and matched. The result is returned.
    * \param image_left the left rectified image
    * \param image_right the right rectified image
    * \param max_y_diff the maximum difference in y for keypoints that match
    * \param max_angle_diff the maximum difference of the keypoint angles 
    *        for keypoints that match
    * \param max_size_diff the maximum difference in size for keypoints that match
    * \return vector of stereo descriptors
    */
    std::vector<StereoFeature>
        extract(const cv::Mat& image_left, const cv::Mat& image_right,
                double max_y_diff = 2.0, double max_angle_diff = 4.0, 
                int max_size_diff = 5);

    /**
    * Computes a match candidate mask that fulfills epipolar constraints, i.e.
    * it is set to 255 for keypoint pairs that should be allowed to match and
    * 0 for all other pairs. Keypoints with different octaves will not be
    * allowed to match.
    * \param keypoints_left keypoints extracted from the left image
    * \param keypoints_right keypoints extracted from the right image
    * \param match_mask matrix to store the result, will be allocated to
    *        rows = keypoints_left.size(), cols = keypoints_right.size()
    *        with type = CV_8UC1.
    * \param max_y_diff the maximum difference of the y coordinates of
    *        left and right keypoints to be accepted as match candidate
    * \param max_angle_diff the maximum difference of the keypoint orientation
    *        in degrees
    * \param max_size_diff the maximum difference of keypoint sizes to accept
    *
    */
    static void computeMatchMask(
            const std::vector<Feature>& features_left,
            const std::vector<Feature>& features_right,
            cv::Mat& match_mask, double max_y_diff,
            double max_angle_diff, int max_size_diff);

    /**
    * Matches two sets of descriptors using cross check, i.e. a match
    * is added for each pair that was matched from left to right AND from
    * right to left.
    * \param features_left descriptors for left image
    * \param features_right descriptors for right image
    * \param matches vector to store matches
    * \param match_mask the mask to use to allow matches, if empty, all
    *        descriptors are matched to each other
    */
    void crossCheckMatching(
            const std::vector<Feature>& features_left, 
            const std::vector<Feature>& features_right,
            std::vector<cv::DMatch>& matches, 
            const cv::Mat& match_mask = cv::Mat());

    /**
    * \param feature_extractor new feature extractor
    */
    void setFeatureExtractor(FeatureExtractor::Ptr& feature_extractor);

  private:

    FeatureExtractor::Ptr feature_extractor_;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;

};

/**
* Paints the result of a stereo keypoint extractor
*/
void paintStereoFeatureMatchings(cv::Mat& image, const cv::Mat& image_left, 
        const cv::Mat& image_right, 
        const std::vector<StereoFeature>& stereo_features);

} // end of namespace object_detection

#endif // defined STEREO_FEATURE_EXTRACTOR


