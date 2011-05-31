#ifndef FEATURE_EXTRACTOR_FACTORY_H
#define FEATURE_EXTRACTOR_FACTORY_H

#include "feature_extractor.h"

namespace object_detection
{

/**
* \class FeatureExtractorFactory
*/
class FeatureExtractorFactory
{

  public:

    /**
    * Factory method
    */
    static FeatureExtractor::Ptr create(const std::string& name);
};

} // end of namespace object_detection

#endif // defined FEATURE_EXTRACTOR_FACTORY_H


