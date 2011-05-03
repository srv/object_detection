
#include "colored_parts_classifier.h"

#include <highgui.h>


using object_detection::ColoredPartsClassifier;

ColoredPartsClassifier::ColoredPartsClassifier() :
    num_hue_bins_(DEFAULT_NUM_HUE_BINS),
    num_saturation_bins_(DEFAULT_NUM_SATURATION_BINS),
    min_saturation_(DEFAULT_MIN_SATURATION)
{
}

cv::Mat ColoredPartsClassifier::preprocessImage(const cv::Mat& image)
{
    // convert image to hsv
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, CV_BGR2HSV);
    return hsv_image;
}

cv::MatND ColoredPartsClassifier::computeHistogram(const cv::Mat& image, 
            const cv::Mat& mask) const
{
    // we assume that the image is a regular
    // three channel image
    CV_Assert(image.type() == CV_8UC3);

    cv::Mat preprocessed_image = preprocessImage(image);
    //return calculateHistogram(preprocessed_image, num_hue_bins_,
    //        num_saturation_bins_, mask);

    // dimensions of the histogram
    int histogram_size[] = {num_hue_bins_, num_saturation_bins_};

    // ranges for the histogram
    float hue_ranges[] = {0, 180};
    float saturation_ranges[] = {min_saturation_, 256};
    const float* ranges[] = {hue_ranges, saturation_ranges};

    // channels for wich to compute the histogram (H and S)
    int channels[] = {0, 1};

    cv::MatND histogram;

    // calculation
    int num_arrays = 1;
    int dimensions = 2;
    cv::calcHist(&preprocessed_image, num_arrays, channels, mask, histogram, 
            dimensions, histogram_size, ranges);

    return histogram;

}

cv::Mat ColoredPartsClassifier::backprojectHistogram(const cv::MatND& histogram,
            const cv::Mat& image) const
{
    // we assume that the image is a regular
    // three channel image
    CV_Assert(image.type() == CV_8UC3);

    cv::Mat preprocessed_image = preprocessImage(image);

    // channels for wich to compute the histogram (H and S)
    int channels[] = {0, 1};

    // ranges for the histogram
    float hue_ranges[] = {0, 180};
    float saturation_ranges[] = {min_saturation_, 256};
    const float* ranges[] = {hue_ranges, saturation_ranges};

    cv::Mat back_projection;
    int num_arrays = 1;
    cv::calcBackProject(&preprocessed_image, num_arrays, channels, histogram,
           back_projection, ranges);

    return back_projection;
}


