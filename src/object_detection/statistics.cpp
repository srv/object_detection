#include "statistics.h"

object_detection::Statistics object_detection::computeStatistics(const cv::Mat& image, const cv::Mat& mask)
{
    cv::Scalar mean, stddev;
    cv::meanStdDev(image, mean, stddev, mask);
    object_detection::Statistics statistics;
    statistics.mean = mean[0];
    statistics.stddev = stddev[0];
    // apply mask
    cv::Mat masked;
    if (!mask.empty())
    {
        image.copyTo(masked, mask);
    }
    else
    {
        masked = image;
    }
    cv::Moments moments = cv::moments(masked);

    statistics.center_of_mass.x = moments.m10 / moments.m00;
    statistics.center_of_mass.y = moments.m01 / moments.m00;

    statistics.area = moments.m00;

    double mu_diff = moments.mu20 - moments.mu02;
    if (mu_diff == 0)
    {
        statistics.main_axis_angle = 0.0;
    }
    else
    {
        statistics.main_axis_angle = atan2(2*moments.mu11, mu_diff) / 2.0;
    }

    return statistics;
}

std::vector<double> object_detection::computeGaussIntersections(
        double m1, double s1, double m2, double s2)
{
    std::vector<double> intersections;
    if (s1 != s2)
    {
        double s1_2 = s1 * s1;
        double s2_2 = s2 * s2;
        double m1_2 = m1 * m1;
        double m2_2 = m2 * m2;
        double p = 2.0 * (m2 * s1_2 - m1 * s2_2) / (s2_2 - s1_2);
        double q = (m1_2 * s2_2 - m2_2 * s1_2 - 2.0 * log(s2/s1) * s1_2 * s2_2) / (s2_2 - s1_2);
        double x1 = -p/2.0 + sqrt(p*p/4 - q);
        double x2 = -p/2.0 - sqrt(p*p/4 - q);
        intersections.resize(2);
        intersections[0] = x1;
        intersections[1] = x2;
    }
    else
    {
        if (m1 == m2)
        {
            // curves are identical
        }
        else
        {
            intersections.resize(1);
            intersections[0] = (m1 + m2) / 2.0;
        }
    }
    return intersections;
}



std::ostream& operator<< (std::ostream& out, const object_detection::Statistics& statistics)
{
    out << "mean: " << statistics.mean << " stddev: " << statistics.stddev
        << " center of mass: (" << statistics.center_of_mass.x << ","
        << statistics.center_of_mass.y << ")" 
        << " main axis angle: " << statistics.main_axis_angle
        << " area: " << statistics.area;
    return out;
}
