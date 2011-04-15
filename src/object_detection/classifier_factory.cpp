#include "classifier_factory.h"
#include "color_classifier.h"

using boost::shared_ptr;
using object_detection::Classifier;
using object_detection::ColorClassifier;
using object_detection::ClassifierFactory;

shared_ptr<Classifier> ClassifierFactory::create(const std::string& name)
{
    if (name == "ColorClassifier")
    {
        shared_ptr<Classifier> op(new ColorClassifier());
        return op;
    }
    else
    {
        shared_ptr<Classifier> op;
        return op;
    }
}

