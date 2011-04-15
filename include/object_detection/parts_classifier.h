#ifndef PARTS_CLASSIFIER_H
#define PARTS_CLASSIFIER_H

#include <string>
#include <vector>

#include <cv.h>

namespace object_detection {

/**
 * \class PartsClassifier
 * \author Stephan Wirth
 * \brief Interface for classifiers that train and classify parts of an object.
 * A parts classifier takes an image together with a binary image as training
 * input.
 * It learns two classes: object parts and background by identifying those
 * parts of the object that differ very much from the background.
 * For classification it takes an image as input and classifies each pixel as
 * belonging to the trained object parts or to background. 
 * The output is a probability map in which
 * for each pixel the probability belonging to parts of the object is marked.
 */
class PartsClassifier
{
public:

    /**
     * Virtual destructor (empty)
     */
	virtual ~PartsClassifier() {};

	/**
	 * \return name of the PartsClassifier
	 */
	virtual std::string getName() const = 0;

    /**
     * \brief trains the classifier.
     * \param image the input training image
     * \param the input training mask. A value = 0 in the mask means the
     *        corresponding pixel in the image is background, a value != 0 
     *        means the pixel belongs to the object.
     */
    virtual void train(const cv::Mat& image, const cv::Mat& mask) = 0;
    
	/**
	 * \brief Runs the classifier.
	 * \param image input image
     * \param rois array of regions of interest that the classifier should use,
     *        if empty the whole image is classified.
     * \return a probability image where background is denoted as 0 and object
     *         as 1.
	 */
	virtual cv::Mat classify(const cv::Mat& image, 
            const std::vector<cv::Rect>& rois = std::vector<cv::Rect>()) const = 0;
    
};

}


#endif /* PARTS_CLASSIFIER_H */

