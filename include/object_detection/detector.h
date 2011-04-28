#ifndef DETECTOR_H
#define DETECTOR_H

#include <vector>

#include <boost/shared_ptr.hpp>

#include "trainable.h"

namespace cv {
    class Mat;
    template<typename T> class Rect_;
    typedef Rect_<int> Rect;
}

namespace object_detection {

struct Detection;
class ObjectPartsDetector;

/**
 * \class Detector
 * \author Stephan Wirth
 * \brief The object detector.
 * For the detection (usage) of the detector, it takes as input an image
 * and (optionally) some regions
 * that define a search space. The output is a list of detections that contain
 * name and location of detected objects (\see Detection).
 */
class Detector : public Trainable
{
public:

    /**
     * Constructor
     */
    Detector();

    /**
     * Destructor
     */
	~Detector() {};

	/**
	 * \brief Run the object detector.
	 * \param image input image
     * \param rois array of regions of interest that the detector should use
     * \return a list of detections, empty if nothing detected
	 */
    std::vector<Detection> detect(const cv::Mat& image, 
            const std::vector<cv::Rect>& rois = std::vector<cv::Rect>());

    void train(const TrainingData& training_data);

    bool isTrained() const { return is_trained_; };

private:

    // stores if the detector was trained
    bool is_trained_;

    // the set of object part detectors 
    std::vector<boost::shared_ptr<ObjectPartsDetector> > object_parts_detectors_;

    // stores the object outline
    std::vector<cv::Point> centered_object_outline_;

};

}


#endif /* DETECTOR_H */

