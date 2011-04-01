#ifndef INTEREST_OPERATOR_H 
#define INTEREST_OPERATOR_H

#include <string>
#include <vector>

namespace cv {
    class Mat;
    template<typename T> class Rect_;
    typedef Rect_<int> Rect;
}

namespace object_detection {

struct TrainingData;

/**
 * \class InterestOperator
 * \author Stephan Wirth
 * \brief Interface for interest operators.
 * An interest operator takes as input an
 * image and (optionally) some regions
 * that define a search space. The output is a list of regions of interest.
 */
class InterestOperator 
{
public:

    /**
     * Virtual destructor (empty)
     */
	virtual ~InterestOperator() {};

	/**
	 * \return name of the operator
	 */
	virtual std::string getName() const = 0;

	/**
	 * \brief Run the interest operator.
	 * \param image input image
     * \param rois array of regions of interest that the operator should use
     * \return a list of regions of interest for further processing
	 */
	virtual std::vector<cv::Rect> computeRegionsOfInterest(const cv::Mat& image,
            const std::vector<cv::Rect>& rois) = 0;
    
};

}


#endif /* INTEREST_OPERATOR_H */

