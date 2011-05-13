#ifndef TEXTURED_PARTS_CLASSIFIER_H
#define TEXTURED_PARTS_CLASSIFIER_H

#include <cassert>

#include "histogram_based_parts_classifier.h"

namespace object_detection {

/**
 * \class TexturedPartsClassifier
 * \author Stephan Wirth
 * \brief classifies textured parts of an object
 */
class TexturedPartsClassifier : public HistogramBasedPartsClassifier
{
public:

    /**
    * Constructs a new TexturedPartsClassifier with default parameters.
    */
    TexturedPartsClassifier();

    std::string getName() const { return "TexturedPartsClassifier"; }

    cv::MatND computeHistogram(const cv::Mat& image, 
            const cv::Mat& mask) const;

    cv::Mat backprojectHistogram(const cv::MatND& histogram,
            const cv::Mat& image) const;


    inline void setNumBins(int num_bins)
    {
        assert(num_bins > 0 && num_bins <= 256);
        num_bins_ = num_bins;
    }

    inline int numBins() const { return num_bins_; };

private:

    /**
    * performs the necessary preprocessing on the image
    * \param image input image
    * \return preprocessed image
    */
    cv::Mat preprocessImage(const cv::Mat& image) const;


    /// default value for number of bins
    static const int DEFAULT_NUM_BINS = 32;

    /// number of bins,
    /// defaults to DEFAULT_NUM_BINS
    int num_bins_;

};

}


#endif /* TEXTURED_PARTS_CLASSIFIER_H */

