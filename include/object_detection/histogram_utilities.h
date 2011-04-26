#ifndef HISTOGRAM_UTILITIES_H
#define HISTOGRAM_UTILITIES_H

#include <cv.h>

namespace object_detection {
namespace histogram_utilities {

    /**
     * \brief calculates and returns a histogram
     * \param hsv_image input image, must be in HSV color format, 8UC3
     * \param num_hue_bins number of bins to use to sample the hue channel
     * \param num_saturation_bins number of bins to sample the saturation
     *        channel
     * \param mask a mask to use
     * \return histogram of input image (Hue-Saturation-Histogram)
     */
    cv::MatND calculateHistogram(const cv::Mat& hsv_image, int num_hue_bins, 
            int num_saturation_bins, const cv::Mat& mask);

    /**
     * \brief calculates a histogram, accumulates it to a given one.
     * \param hsv_image input image, must be in HSV color format, 8UC3
     * \param num_hue_bins number of bins to use to sample the hue channel
     * \param num_saturation_bins number of bins to sample the saturation
     *        channel
     * \param mask a mask to use
     * \param histogram the output histogram, must be unallocated or size
     *        num_hue_bins * num_saturation_bins
     */
    void accumulateHistogram(const cv::Mat& hsv_image, int num_hue_bins, 
        int num_saturation_bins, const cv::Mat& mask, cv::MatND& histogram);
 
     /**
     * \brief performs a backprojection of a histogram on an image
     * \param histogram the histogram to project, as produced by
     *        calculateHistogram()
     * \param hsv_image image to project the histogram on, must be in HSV
     *        color format, 8UC3
     * \return backprojection of the histogram as single channel array that
     *         has the same dimensions as hsv_image
     */
    cv::Mat calculateBackprojection(const cv::MatND& histogram,
            const cv::Mat& hsv_image);

    /**
     * \brief shows a histogram in an opencv window
     * Calls calculateHistogram followed by the version below of showHSHistogram.
     * \param image the image to show the histogram for
     * \param num_hue_bins number of bins to use to sample the hue channel
     * \param num_saturation_bins number of bins to sample the saturation
     * \param mask a mask
     * \param name the name to use to create the window
     */
    void showHSHistogram(const cv::Mat& image, int num_hue_bins,
            int num_saturation_bins, const cv::Mat& mask, const std::string& name);

    /**
    * \brief paints a nice two-dimensional histogram of hue and saturation
    * \param histogram the histogram, must be a two-channel Hue-Saturation histogram.
    * \param name the name of the window title where the histogram will be shown
    */
    void showHSHistogram(const cv::MatND& histogram,
        const std::string& name);

    /**
    * \brief paints 1D and 2D histograms
    * \param histogram the histogram, must have a one or two-channels.
    * \param name the name of the window title where the histogram will be shown
    */
    void showHistogram(const cv::MatND& histogram,
        const std::string& name);
 

} // namespace histogram_utilities
} // namespace object_detection

#endif

