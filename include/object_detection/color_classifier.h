#ifndef COLOR_CLASSIFIER_H
#define COLOR_CLASSIFIER_H

#include "classifier.h"

namespace object_detection {

/**
 * \class ColorClassifier
 * \author Stephan Wirth
 * \brief Classifies pixels by its color.
 */
class ColorClassifier : public Classifier
{
public:

    /**
     * Constructor
     */
    ColorClassifier();

    /**
     * Destructor
     */
	~ColorClassifier() {};

	std::string getName() const { return "ColorClassifier"; };

    void train(const cv::Mat& image, const cv::Mat& mask);
	
	cv::Mat classify(const cv::Mat& image, 
            const std::vector<cv::Rect>& rois = std::vector<cv::Rect>()) const;

private:

    // stores the histogram of the object
    cv::MatND object_histogram_;

    // stores the histogram of the background
    cv::MatND background_histogram_;

    // stores the histogram of significant colors
    cv::MatND significant_colors_histogram_;

    // stores if the classifier is trained
    bool is_trained_;
    
};

}


#endif /* COLOR_CLASSIFIER_H */

