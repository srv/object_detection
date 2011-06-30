#ifndef DETECTOR_H
#define DETECTOR_H

#include <vector>

#include <boost/shared_ptr.hpp>

#include "trainable.h"
#include "model.h"

namespace cv {
    class Mat;
    template<typename T> class Rect_;
    typedef Rect_<int> Rect;
}

namespace object_detection {

struct Detection;
class ObjectPartsDetector;
class StereoFeature;

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
     * \param config_file_name the name of the config file to use
     */
    Detector(const std::string& config_file_name);

    /**
     * Destructor
     */
	~Detector() {};

	/**
	 * \brief Run the object detector.
	 * \param image input image
	 * \param stereo_features extracted stereo features
     * \param rois array of regions of interest that the detector should use
     * \return a list of detections, empty if nothing detected
	 */
    std::vector<Detection> detect(const cv::Mat& image, 
            const std::vector<StereoFeature>& stereo_features,
            const std::vector<cv::Rect>& rois = std::vector<cv::Rect>());

    void train(const TrainingData& training_data);

    bool isTrained() const { return is_trained_; };

    static bool estimatePose(const Model& object_model,
            const Model& scene_model, cv::Mat& transformation);

private:

    /**
    * Setup routine, reads config files and sets parameters
    */
    void setup();

    // stores if the detector was trained
    bool is_trained_;

    // stores the name of the config file
    std::string config_file_name_;

    // the set of object part detectors 
    std::vector<boost::shared_ptr<ObjectPartsDetector> > object_parts_detectors_;

    // stores the object outline
    std::vector<cv::Point> centered_object_outline_;

    // stores the object model
    Model object_model_;

};

}


#endif /* DETECTOR_H */

