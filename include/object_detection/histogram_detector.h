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


    /**
     * \brief calculates and returns a histogram
     * \param hsv_image input image, must be in HSV color format, 8UC3
     * \param num_hue_bins number of bins to use to sample the hue channel
     * \param num_saturation_bins number of bins to sample the saturation
     *        channel
     * \param mask a mask to use
     * \return histogram of input image (Hue-Saturation-Histogram)
     */
    static cv::MatND calculateHistogram(const cv::Mat& hsv_image,
            int num_hue_bins, int num_saturation_bins, const cv::Mat& mask);
    
    /**
     * \brief performs a backprojection of a histogram on an image
     * \param histogram the histogram to project, as produced by
     *        calculateHistogram()
     * \param hsv_image image to project the histogram on, must be in HSV
     *        color format, 8UC3
     * \return backprojection of the histogram as single channel array that
     *         has the same dimensions as hsv_image
     */
    static cv::Mat calculateBackprojection(const cv::MatND& histogram,
            const cv::Mat& hsv_image);

    /**
     * \brief shows a histogram in an opencv window
     * \param image the image to show the histogram for
     * \param num_hue_bins number of bins to use to sample the hue channel
     * \param num_saturation_bins number of bins to sample the saturation
     * \param mask a mask
     * \param name the name to use to create the window
     */
    static void showHSHistogram(const cv::Mat& image, int num_hue_bins,
            int num_saturation_bins, const cv::Mat& mask, const std::string& name);
    static void showHSHistogram(const cv::MatND& histogram,
        const std::string& name);
 
    // stores the histogram of the training data
    cv::MatND object_histogram_;

    // stores the object size
    cv::Size object_size_;

    // stores if the detector has been trained
    bool is_trained_;
};

}


#endif /* HISTOGRAM_DETECTOR_H */
