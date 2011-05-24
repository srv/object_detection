#include "stereo_keypoint_extractor.h"

StereoKeypointExtractor::extract(const cv::Mat& image_left, const cv::Mat& image_right)
{
    std::vector<KeyPoint> keypoints_left = extractKeypoints(image_left);
    // find matching keypoints in right image
    std::vector<DMatch> matchings = findMatchingKeypoints(keypoints_left, image_right);
    // or
    std::vector<KeyPoint> keypoints_right = extractKeypoints(image_right);
    std::vector<DMatch> matchings = stereoMatch(keypoints_left, keypoints_right);

    std::vector<cv::Point3d> world_points(matchings.size());
    for (size_t i = 0; i < matchings.size(); ++i)
    {
        int index_left = matchings[i].queryIdx;
        int index_right = matchings[i].trainIdx;
        const KeyPoint& keypoint_left = keypoints_left[index_left];
        const KeyPoint& keypoint_right = keypoints_right[index_right];
        double disparity = keypoint_left.pt.x - keypoint_right.pt.x;
        stereo_camera_model_.projectDisparityTo3d(keypoint_left.pt, disparity, world_points[i]);
    }
}
