#ifndef OBJECT_PARTS_DETECTOR_H
#define OBJECT_PARTS_DETECTOR_H

#include <vector>
#include <boost/shared_ptr.hpp>

#include "statistics.h"

namespace cv {
    class Mat;
    template<typename T> class Rect_;
    typedef Rect_<int> Rect;
}

namespace object_detection {

struct Detection;
class PartsClassifier;

/**
 * \class ObjectPartsDetector
 * \author Stephan Wirth
 * \brief Detector for object parts.
 * Each ObjectPartsDetector uses one PartsClassifier to learn significant
 * parts of the object that differ from the background.
 * ObjectPartsDetector stores the shapes that define the object parts
 * and matches them with detected shapes to compute a detection.
 */
class ObjectPartsDetector
{
public:

    /**
     * Constructs a new ObjectPartsDetector.
     * \param parts_classifier_ the classifier to use. If the pointer is invalid
     *        an exception is raised.
     */
    ObjectPartsDetector(boost::shared_ptr<PartsClassifier> parts_classifier);

    /**
     * Destructor
     */
	~ObjectPartsDetector() {};

    /**
    * \brief trains the ObjectPartsDetector.
    *  The ObjectPartsDetector runs its ObjectPartsLearner and stores those parts
    *  of the object as shapes that the learner can detect.
    */
    void train(const cv::Mat& image, const cv::Mat& object_mask);

	/**
	 * \brief Run the object parts detector.
	 * \param image input image
     * \param rois array of regions of interest that the detector should use
     * \return a list of detections, empty if nothing detected
	 */
    std::vector<Detection> detect(const cv::Mat& image);

private:

    /**
    * Computes the best threshold to separate foreground and background
    * \param image the input image (must be one channel grayscale)
    * \param mask the mask that masks foreground
    */
    double computeBestThreshold(const cv::Mat& image, const cv::Mat& mask);

    /**
    * \param image the input image (must be one channel binary image)
    * \return the shapes of the input image
    */
    std::vector<std::vector<cv::Point> > extractShapes(const cv::Mat& image);

    /**
    * Sorts given shapes according to their size and returns the set that has
    * size bigger than half of the biggest size.
    * \param shapes input shapes
    * \return the big shapes of the input shapes
    */
    static std::vector<std::vector<cv::Point> > getBiggestShapes(
        const std::vector<std::vector<cv::Point> >& shapes);

    /// stores the classifier for the parts
    boost::shared_ptr<PartsClassifier> parts_classifier_;

    /// stores the threshold to apply to the output of the classifier
    double threshold_;

    /// the parts of the object that the detector can detect
    std::vector<std::vector<cv::Point> > object_part_shapes_;

    /// statistics of the object part shapes
    Statistics object_part_statistics_;

    /// statistics of the full object
    Statistics full_object_statistics_;

};

}


#endif /* OBJECT_PARTS_DETECTOR_H */

