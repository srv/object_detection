#ifndef COLORED_PARTS_CLASSIFIER_H
#define COLORED_PARTS_CLASSIFIER_H

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


    std::string getName() const { return "ColoredPartsClassifier"; }

    cv::MatND computeHistogram(const cv::Mat& image, 
            const cv::Mat& mask) const;

    cv::Mat backprojectHistogram(const cv::MatND& histogram,
            const cv::Mat& image) const;

private:

    /**
    * performs the necessary preprocessing on the image
    * \param image input image
    * \return preprocessed image
    */
    static cv::Mat preprocessImage(const cv::Mat& image);


};

}


#endif /* COLORED_PARTS_CLASSIFIER_H */

