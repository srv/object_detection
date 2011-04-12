#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>

#include <cv.h>

namespace object_detection {

/**
 * \class Classifier
 * \author Stephan Wirth
 * \brief Interface for classifier.
 * A classifier takes an image together with a binary image as training input.
 * It learns two classes: object and background.
 * For classification it takes an image as input and classifies each pixel as
 * belonging to object or to background. The output is a probability map in which
 * for each pixel the probability belonging to the object is marked.
 */
class Classifier
{
public:

    /**
     * Virtual destructor (empty)
     */
	virtual ~Classifier() {};

	/**
	 * \return name of the Classifier
	 */
	virtual std::string getName() const = 0;

    /**
     * \brief trains the classifier.
     * \param images the input training images
     * \param the input training masks. A value = 0 in the mask means the
     *        corresponding pixel in the image is background, a value != 0 
     *        means the pixel belongs to the object.
     * images.size() must be equal to masks.size()
     * \return true if training successful, false otherwise.
     */
    virtual bool train(const std::vector<cv::Mat>& images,
                       const std::vector<cv::Mat>& masks) = 0;
    /**
     * \brief trains the classifier.
     * Provided for convenience, calls the above method.
     * \param image the input training image
     * \param the input training mask. A value = 0 in the mask means the
     *        corresponding pixel in the image is background, a value != 0 
     *        means the pixel belongs to the object.
     * \return true if training successful, false otherwise.
     */
    bool train(const cv::Mat& image,
                       const cv::Mat& mask) {
        std::vector<cv::Mat> images(1);
        images[0] = image;
        std::vector<cv::Mat> masks(1);
        masks[0] = mask;
        return train(images, masks);
    }
    
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


#endif /* CLASSIFIER_H */

