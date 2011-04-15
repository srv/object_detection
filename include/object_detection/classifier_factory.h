#ifndef CLASSIFIER_FACTORY_H 
#define CLASSIFIER_FACTORY_H

#include <string>
#include <boost/shared_ptr.hpp>

#include "classifer.h"

namespace object_detection {

/**
 * \class ClassifierFactory
 * \author Stephan Wirth
 * \brief Factory for Classifiers.
 */
class ClassifierFactory
{
public:

    /**
    * Creates and returns a new classifer by given name. If no
    * classifier with the given name is known,
    * a NULL-pointer is returned (i.e. ptr.get() == NULL).
    */
    static boost::shared_ptr<Classifier> create(const std::string& name);
    
};

}


#endif /* CLASSIFIER_FACTORY_H */

