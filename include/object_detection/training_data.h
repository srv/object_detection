#ifndef TRAININGDATA_H 
#define TRAININGDATA_H 

#include <vector>

namespace cv {
    class Mat;
    template<typename T> class Point_;
    typedef Point_<int> Point;
}

namespace object_detection {

struct StereoFeature
{
    cv::KeyPoint key_point;
    cv::Point3f world_point;
    vector<float> descriptor;
};

/**
 * \struct TrainingData
 * \author Stephan Wirth
 * \brief Data structure for training data that is used by Detectors.
 */
struct TrainingData 
{
    /// the image on which the object is visible
    cv::Mat image;

    /// The polygon that describes the object outline in the image.
    /// The last point is connected to the first one.
    std::vector<cv::Point> object_outline;

    /// The set of stereo features
    std::vector<StereoFeature> stereo_features;

    /**
     * \return true if the data is valid, false otherwise
     */
    bool isValid() const
    {
        return (image.cols != 0 && image.rows != 0 &&
                object_outline.size() > 2);
    }
};

}


#endif /* TRAININGDATA_H */

