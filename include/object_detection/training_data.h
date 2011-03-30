#ifndef TRAININGDATA_H 
#define TRAININGDATA_H 

namespace cv {
    class Mat;
    class RotatedRect;
}

namespace object_detection {

/**
 * \struct TrainingData
 * \author Stephan Wirth
 * \brief Data structure for training data that is used by Detectors.
 * The training data consists of an image and a rotated rectangle that describes
 * the location of the object that has to be detected.
 */
struct TrainingData 
{
    /// the image on which the object is visible
    cv::Mat image;

    /// the region that describes the exact location of the object in the image
    cv::RotatedRect bounding_rotated_rect;

    /**
     * \return true if the data is valid (image and rect dimensions != 0),
     * false otherwise
     */
    bool isValid() const
    {
        return (image.cols != 0 && image.rows != 0 &&
                bounding_rotated_rect.size.width != 0 &&
                bounding_rotated_rect.size.height != 0);
    }
};

}


#endif /* TRAININGDATA_H */

