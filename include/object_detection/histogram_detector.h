#ifndef HISTOGRAM_DETECTOR_H
#define HISTOGRAM_DETECTOR_H

#include <cv.h>

#include "detector.h" 

namespace object_detection {

/**
 * \class HistogramDetector
 * \author Stephan Wirth
 * \brief A simple detector using histogram backprojection.
 * In the training phase, the HistogramDetector computes and saves a histogram
 * of the object it has to detect. In the detection phase, this histogram is
 * backprojected on the given image which results in a probability map.
 * High peaks in this map mean high probability of the object at the location
 * of the peak.
 */
class HistogramDetector : public Detector
{
public:

    /**
     * \brief Constructor 
     */
    HistogramDetector(); 

    /**
     * \brief Destructor
     */
	~HistogramDetector() {}

	std::string getName() const { return "HistogramDetector"; }

    void train(const TrainingData& training_data);

    std::vector<Detection> detect(const cv::Mat& image, 
            const std::vector<cv::Rect>& rois = std::vector<cv::Rect>());

    bool isTrained() const { return is_trained_; };

private:

    // stores the histogram of the training data
    cv::MatND object_histogram_;

    // stores the object size
    cv::Size object_size_;

    // stores if the detector has been trained
    bool is_trained_;
};

}


#endif /* HISTOGRAM_DETECTOR_H */
