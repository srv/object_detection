#include <cv.h>
#include <highgui.h>

#include "histogram_utilities.h"

namespace object_detection
{
namespace histogram_utilities
{

// calculates a two dimensional hue-saturation histogram
cv::MatND calculateHistogram(const cv::Mat& hsv_image, int num_hue_bins, 
        int num_saturation_bins, const cv::Mat& mask)
{   
    // we assume that the image is a regular
    // three channel image
    CV_Assert(hsv_image.type() == CV_8UC3);

    // dimensions of the histogram
    int histogram_size[] = {num_hue_bins, num_saturation_bins};

    // ranges for the histogram
    float hue_ranges[] = {0, 180};
    float saturation_ranges[] = {0, 256};
    const float* ranges[] = {hue_ranges, saturation_ranges};

    // channels for wich to compute the histogram (H and S)
    int channels[] = {0, 1};

    cv::MatND histogram;

    // calculation
    int num_arrays = 1;
    int dimensions = 2;
    cv::calcHist(&hsv_image, num_arrays, channels, mask, histogram, dimensions,
            histogram_size, ranges);

    return histogram;
}

// calculates a two dimensional hue-saturation histogram and accumulates
void accumulateHistogram(const cv::Mat& hsv_image, int num_hue_bins, 
        int num_saturation_bins, const cv::Mat& mask, cv::MatND& histogram)
{   
    // we assume that the image is a regular
    // three channel image
    CV_Assert(hsv_image.type() == CV_8UC3);

    // dimensions of the histogram
    int histogram_size[] = {num_hue_bins, num_saturation_bins};

    // ranges for the histogram
    float hue_ranges[] = {0, 180};
    float saturation_ranges[] = {0, 256};
    const float* ranges[] = {hue_ranges, saturation_ranges};

    // channels for wich to compute the histogram (H and S)
    int channels[] = {0, 1};

    // calculation
    int num_arrays = 1;
    int dimensions = 2;
    bool uniform = true;
    bool accumulate = true;
    cv::calcHist(&hsv_image, num_arrays, channels, mask, histogram, dimensions,
            histogram_size, ranges, uniform, accumulate);
}


cv::Mat calculateBackprojection(const cv::MatND& histogram,
        const cv::Mat& hsv_image)
{
    // we assume that the image is a regular
    // three channel image
    CV_Assert(hsv_image.type() == CV_8UC3);

    // channels for wich to compute the histogram (H and S)
    int channels[] = {0, 1};

    // ranges for the histogram
    float hue_ranges[] = {0, 180};
    float saturation_ranges[] = {0, 256};
    const float* ranges[] = {hue_ranges, saturation_ranges};

    cv::Mat back_projection;
    int num_arrays = 1;
    cv::calcBackProject(&hsv_image, num_arrays, channels, histogram,
           back_projection, ranges);

    return back_projection;
}


void showHSHistogram(const cv::Mat& image,
        int num_hue_bins, int num_saturation_bins, const cv::Mat& mask,
        const std::string& name)
{
    // we assume that the image is a regular
    // three channel image
    CV_Assert(image.type() == CV_8UC3);
    
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, CV_BGR2HSV);

    cv::MatND histogram = calculateHistogram(hsv_image, num_hue_bins,
            num_saturation_bins, mask);
    showHSHistogram(histogram, name);
}
 
void showHSHistogram(const cv::MatND& histogram,
        const std::string& name)
{
    int num_hue_bins = histogram.rows;
    int num_saturation_bins = histogram.cols;

    // visualization
    double max_value = 0;
    cv::minMaxLoc(histogram, 0, &max_value, 0, 0);
    int scale = 8;
    cv::Mat histogram_image_hsv = 
        cv::Mat::zeros((num_saturation_bins + 1) * scale, (num_hue_bins + 1) * scale, CV_8UC3);

    // paint x axis
    for( int h = 1; h < num_hue_bins + 1; h++ )
        cv::rectangle( histogram_image_hsv, cv::Point(h*scale, 0),
                cv::Point((h + 1)*scale - 1, scale - 1),
                cv::Scalar(1.0 * (h - 1) / num_hue_bins * 180.0, 255, 255, 0),
                CV_FILLED);

    // paint y axis
    for( int s = 1; s < num_saturation_bins + 1; s++ )
        cv::rectangle( histogram_image_hsv, cv::Point(0, s * scale),
                cv::Point(scale - 1, (s + 1)*scale - 1),
                cv::Scalar(180, 1.0 * (s - 1) / num_saturation_bins * 255.0, 255, 0),
                CV_FILLED);

    cv::Mat histogram_image;
    cv::cvtColor(histogram_image_hsv, histogram_image, CV_HSV2BGR);

    for( int h = 1; h < num_hue_bins + 1; h++ )
        for( int s = 1; s < num_saturation_bins + 1; s++ )
        {
            float binVal = histogram.at<float>(h - 1, s - 1);
            int intensity = cvRound(binVal * 255 / max_value);
            cv::rectangle( histogram_image, cv::Point(h*scale, s*scale),
                cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                cv::Scalar::all(intensity),
                CV_FILLED );
         }

    cv::namedWindow( name );
    cv::imshow( name, histogram_image );
}

void showHistogram(const cv::MatND& histogram,
        const std::string& name)
{
    assert(histogram.dims == 1 || histogram.dims == 2);
    assert(histogram.type() == CV_32FC1);

    double max_value = 0;
    cv::minMaxLoc(histogram, 0, &max_value, 0, 0);
 
    cv::Mat histogram_image;

    if (histogram.dims == 1)
    {
        int x_scale = 4;
        int y_scale = 100;
        histogram_image.create(y_scale, histogram.size[0] * x_scale, CV_8UC1);
        histogram_image = cv::Scalar(0);
        for (int x = 0; x < histogram.size[0]; ++x)
        {
            float binVal = histogram.at<float>(x);
            int height = cvRound(binVal * y_scale / max_value);
            cv::rectangle(histogram_image, cv::Point(x*x_scale, y_scale-1),
                cv::Point((x+1)*x_scale-1, y_scale-height),
                cv::Scalar::all(255),
                CV_FILLED );
        }
    }
    else if (histogram.dims == 2)
    {
        int scale = 8; // pixel size for a bin
        histogram_image.create(histogram.size[0] * scale, histogram.size[1] * scale, CV_8UC1);
        histogram_image = cv::Scalar(0);
        for( int y = 0; y < histogram.rows; y++ )
            for( int x = 0; x < histogram.cols; x++ )
            {
                float binVal = histogram.at<float>(y, x);
                int intensity = cvRound(binVal * 255 / max_value);
                cv::rectangle( histogram_image, cv::Point(x*scale, y*scale),
                    cv::Point( (x+1)*scale - 1, (y+1)*scale - 1),
                    cv::Scalar::all(intensity),
                    CV_FILLED );
            }
    }

    cv::namedWindow( name );
    cv::imshow( name, histogram_image );
}

} // namespace object_detection
} // namespace histogram_utilities

