#ifndef TRAININGDATA_H 
#define TRAININGDATA_H 

#include <vector>

namespace cv {
    class Mat;
    class RotatedRect;
    template<typename T> class Point_;
    typedef Point_<int> Point;
}

namespace object_detection {

/**
 * \struct TrainingData
 * \author Stephan Wirth
 * \brief Data structure for training data that is used by Detectors.
 * The training data consists of an image and an outline of an object of 
 * interest.
 */
struct TrainingData 
{
    /// the image on which the object is visible
    cv::Mat image;

    /// The polygon that describes the object outline in the image.
    /// The last point is connected to the first one.
    std::vector<cv::Point> object_outline;

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

