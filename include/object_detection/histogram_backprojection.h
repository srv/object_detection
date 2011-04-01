#ifndef HISTOGRAM_BACKPROJECTION_H
#define HISTOGRAM_BACKPROJECTION_H

#include <cv.h>

#include "interest_operator.h" 
#include "trainable.h"

namespace object_detection {

/**
 * \class HistogramBackprojection
 * \author Stephan Wirth
 * \brief An interest operator that uses histogram backprojection.
 * In the training phase, the HistogramBackprojection computes and saves a
 * histogram of the object of interest. In the operation phase, this histogram
 * is backprojected on the given image which results in a probability map.
 * High peaks in this map mean high probability of the object at the location
 * of the peak.
 * Some postprocessing using thresholds result regions of interest for the
 * object of interest.
 */
class HistogramBackprojection : public InterestOperator, public Trainable
{
public:

    /**
     * \brief Constructor 
     */
    HistogramBackprojection(); 

    /**
     * \brief Destructor
     */
	~HistogramBackprojection() {}

	std::string getName() const { return "HistogramBackprojection"; }

    void train(const TrainingData& training_data);

    std::vector<cv::Rect> computeRegionsOfInterest(const cv::Mat& image, 
            const std::vector<cv::Rect>& rois = std::vector<cv::Rect>());

    bool isTrained() const { return is_trained_; };

private:

    // stores the histogram of the training data
    cv::MatND object_histogram_;

    // stores if train has been run
    bool is_trained_;
};

}


#endif /* HISTOGRAM_BACKPROJECTION_H */
