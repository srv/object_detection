#ifndef HISTOGRAM_BASED_PARTS_CLASSIFIER_H
#define HISTOGRAM_BASED_PARTS_CLASSIFIER_H

#include <vector>

#include <cv.h>

#include "parts_classifier.h"

namespace object_detection {

/**
 * \class HistogramBasedPartsClassifier
 * \author Stephan Wirth
 * \brief Base class for histogram based parts classifiers.
 */
class HistogramBasedPartsClassifier : public PartsClassifier
{
public:

    /**
     * Virtual destructor (empty)
     */
	virtual ~HistogramBasedPartsClassifier() {};

    void train(const cv::Mat& image, const cv::Mat& mask);
    
	cv::Mat classify(const cv::Mat& image, 
            const std::vector<cv::Rect>& rois = std::vector<cv::Rect>()) const;

    virtual cv::MatND computeHistogram(const cv::Mat& image, 
            const cv::Mat& mask) const = 0;

    virtual cv::Mat backprojectHistogram(const cv::MatND& histogram,
            const cv::Mat& image) const = 0;

private:

    /// stores the histogram of significant elements that is used for 
    /// backprojection
    cv::MatND significant_elements_histogram_;
    
};

}


#endif /* HISTOGRAM_BASED_PARTS_CLASSIFIER_H */

