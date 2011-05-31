#include "feature_extractor_factory.h"

#include "surf_extractor.h"

using namespace object_detection;

FeatureExtractor::Ptr FeatureExtractorFactory::create(const std::string& name)
{
    if (name == "SURF")
    {
        return FeatureExtractor::Ptr(new SurfExtractor());
    }
    else
    {
        FeatureExtractor::Ptr ptr;
        return ptr;
    }
}

