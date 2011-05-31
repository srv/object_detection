#ifndef SURF_EXTRACTOR_H
#define SURF_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>
#include "feature_extractor.h"

namespace object_detection
{

/**
* \class SurfExtractor
*/
class SurfExtractor : public FeatureExtractor
{

  public:
    /**
    * Constructs a surf extractor with default parameters.
    */
    SurfExtractor();

    std::vector<Feature> extract(const cv::Mat& image);

  private:

    // holds opencv's surf method
    cv::SURF surf_;
};

} // end of namespace object_detection

#endif // defined SURF_EXTRACTOR_H

