#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <boost/shared_ptr.hpp>

#include "feature.h"

namespace object_detection
{


/**
* \class FeatureExtractor
* \brief Common interface for feature extractors
*/
class FeatureExtractor
{

  public:
    /**
    * Virtual destructor
    */
    virtual ~FeatureExtractor() {};

    /**
    * Extraction interface.
    * \param image the input image.
    * \return the features extracted.
    */
    virtual std::vector<Feature> extract(const cv::Mat& image) = 0;

    /// Ptr type
    typedef boost::shared_ptr<FeatureExtractor> Ptr;
};

} // end of namespace object_detection

#endif // defined FEATURE_EXTRACTOR_H


