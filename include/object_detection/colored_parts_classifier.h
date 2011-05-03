#ifndef COLORED_PARTS_CLASSIFIER_H
#define COLORED_PARTS_CLASSIFIER_H

#include <cassert>

#include "histogram_based_parts_classifier.h"

namespace object_detection {

/**
 * \class ColoredPartsClassifier
 * \author Stephan Wirth
 * \brief classifies colored parts of an object
 */
class ColoredPartsClassifier : public HistogramBasedPartsClassifier
{
public:

    /**
    * Constructs a new ColoredPartsClassifier with default parameters.
    */
    ColoredPartsClassifier();

    std::string getName() const { return "ColoredPartsClassifier"; }

    cv::MatND computeHistogram(const cv::Mat& image, 
            const cv::Mat& mask) const;

    cv::Mat backprojectHistogram(const cv::MatND& histogram,
            const cv::Mat& image) const;


    inline void setNumHueBins(int num_hue_bins)
    {
        assert(num_hue_bins > 0 && num_hue_bins <= 180);
        num_hue_bins_ = num_hue_bins;
    }

    inline int numHueBins() const { return num_hue_bins_; };

    inline void setNumSaturationBins(int num_saturation_bins)
    {
        assert(num_saturation_bins > 0 && num_saturation_bins < 256);
        num_saturation_bins_ = num_saturation_bins;
    }

    inline int numSaturationBins() const { return num_saturation_bins_; }

    inline void setMinSaturation(int min_saturation)
    {
        assert(min_saturation >= 0 && min_saturation < 256);
        min_saturation_ = min_saturation;
    }

    inline int minSaturation() const { return min_saturation_; }
    
private:

    /**
    * performs the necessary preprocessing on the image
    * \param image input image
    * \return preprocessed image
    */
    static cv::Mat preprocessImage(const cv::Mat& image);


    /// default value for number of hue bins
    static const int DEFAULT_NUM_HUE_BINS = 32;

    /// default value for number of saturation bins
    static const int DEFAULT_NUM_SATURATION_BINS = 16;

    /// default value for minimum saturation
    static const int DEFAULT_MIN_SATURATION = 0;

    /// number of bins for the hue channels,
    /// defaults to DEFAULT_NUM_HUE_BINS
    int num_hue_bins_;

    /// number of bins for the saturation channel, defaults to
    /// DEFAULT_NUM_SATURATION_BINS
    int num_saturation_bins_;

    /// minimum saturation that has to have a color to be
    /// part of the object model, defaults to DEFAULT_MIN_SATURATION
    int min_saturation_;

};

}


#endif /* COLORED_PARTS_CLASSIFIER_H */

