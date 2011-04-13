#ifndef STATISTICS_H
#define STATISTICS_H

#include <iostream>
#include <vector>

#include <cv.h>


namespace object_detection {

/**
 * \struct Statistics
 */
struct Statistics
{
    /// mean value
    double mean;

    /// standard deviation
    double stddev;

    /// center of mass
    cv::Point center_of_mass;

    /// main axis angle (computed from moments)
    double main_axis_angle;

    /// area (computed from moments)
    double area;
};
    
/**
 * \brief computes statistics on the first channel of given image
 * \param image input image
 * \param mask input mask
 * \return statistics
 */
Statistics computeStatistics(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

/**
 * \brief computes x coordinates of intersection points of two gauss graphs
 * \param m1 mean of graph 1
 * \param s1 standard deviation of graph 1
 * \param m2 mean of graph 2
 * \param s2 standard deviation of graph 2
 * \return x values of intersection points (maximum two), 
 *         no points if graphs are identical.
 */
std::vector<double> computeGaussIntersections(
        double m1, double s1, double m2, double s2);

} // namespace object_detection


std::ostream& operator<< (std::ostream& out, const object_detection::Statistics& statistics);

#endif

