#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <vector>

namespace cv {
    class Mat;
    template<typename T> class Rect_;
    typedef Rect_<int> Rect;
}

namespace object_detection {

struct TrainingData;
struct Detection;

/**
 * \class Detector
 * \author Stephan Wirth
 * \brief Interface for object detectors.
 * Before a detector can be used, it has to be trained using the
 * method train(). 
 * For the detection (usage) of the detector, it takes as input an image
 * and (optionally) some regions
 * that define a search space. The output is a list of detections that contain
 * name and location of detected objects (\see Detection).
 */
class Detector
{
public:

    /**
     * Virtual destructor (empty)
     */
	virtual ~Detector() {};

	/**
	 * \return name of the detector
	 */
	virtual std::string getName() const = 0;

    /**
     * \brief train the detector
     * \param training_data the data that is used for training 
     */
    virtual void train(const TrainingData& training_data) = 0;

	/**
	 * \brief Run the object detector.
	 * \param image input image
     * \param rois array of regions of interest that the detector should use
     * \return a list of detections, empty if nothing detected
	 */
	virtual std::vector<Detection> detect(const cv::Mat& image, 
            const std::vector<cv::Rect>& rois) = 0;
    
    /**
     * \brief checks if the detector is ready for detection
     * \return true if the detector is trained and can be used calling detect().
     *         don't call detect() when this method returns false.
     */
    virtual bool isTrained() const = 0;
};

}


#endif /* DETECTOR_H */

