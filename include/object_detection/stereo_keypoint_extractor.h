#ifndef STEREO_KEYPOINT_EXTRACTOR_H
#define STEREO_KEYPOINT_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>



namespace object_detection
{

/**
* Struct to hold a stereo matching result
*/
struct StereoDescriptor
{
    cv::Mat descriptor_left;
    cv::Mat descriptor_right;
    cv::KeyPoint keypoint_left;
    cv::KeyPoint keypoint_right;
};


/**
* \class StereoKeypointExtractor
* \brief Extractor for matching keypoints in a rectified stereo image pair
*/
class StereoKeypointExtractor
{

  public:

    /**
    * Constructor creates default feature detector, descriptor extractor
    * and descriptor matcher
    */
    StereoKeypointExtractor();

    /**
    * Extracts stereo keypoints from given rectified stereo image pair.
    * Keypoints for each image are computed, a match mask that preserves
    * the epipolar constraints is computed, descriptors for
    * keypoints are computed and matched. The result is returned.
    * \param image_left the left rectified image
    * \param image_right the right rectified image
    * \return vector of stereo descriptors
    */
    std::vector<StereoDescriptor>
        extract(const cv::Mat& image_left, const cv::Mat& image_right);

    /**
    * Computes a match candidate mask that fulfills epipolar constraints, i.e.
    * it is set to 255 for keypoint pairs that have a y-distance less than
    * a given distance and to 0 for all other pairs.
    * \param keypoints_left keypoints extracted from the left image
    * \param keypoints_right keypoints extracted from the right image
    * \param match_mask matrix to store the result, will be allocated to
    *        rows = keypoints_left.size(), cols = keypoints_right.size()
    *        with type = CV_8UC1.
    * \param max_y_diff the maximal difference of the y coordinates of
    *        left and right keypoints to be accepted as match candidate
    * \param max_angle_diff the maximal difference of the keypoint orientation
    *        in degrees
    */
    void computeMatchMask(
            const std::vector<cv::KeyPoint>& keypoints_left,
            const std::vector<cv::KeyPoint>& keypoints_right,
            cv::Mat& match_mask, double max_y_diff = 2.0,
            double max_angle_diff = 5.0);

    /**
    * Matches two sets of descriptors using cross check, i.e. a match
    * is added for each pair that was matched from left to right AND from
    * right to left.
    * \param descriptors_left descriptors for left image
    * \param descriptors_right descriptors for right image
    * \param matches vector to store matches
    * \param match_mask the mask to use to allow matches, if empty, all
    *        descriptors are matched to each other
    */
    void crossCheckMatching(
            const cv::Mat& descriptors_left, 
            const cv::Mat& descriptors_right,
            std::vector<cv::DMatch>& matches, 
            const cv::Mat& match_mask = cv::Mat());

  private:

    cv::Ptr<cv::FeatureDetector> feature_detector_;
    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor_;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;

};

/**
* Paints the result of a stereo keypoint extractor
*/
void paintStereoDescriptorMatchings(cv::Mat& image, const cv::Mat& image_left, 
        const cv::Mat& image_right, 
        const std::vector<StereoDescriptor>& stereo_descriptors);

} // end of namespace object_detection

#endif // defined STEREO_KEYPOINT_EXTRACTOR


