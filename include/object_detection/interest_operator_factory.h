#ifndef INTEREST_OPERATOR_FACTORY_H 
#define INTEREST_OPERATOR_FACTORY_H

#include <string>
#include <boost/shared_ptr.hpp>

#include "interest_operator.h"

namespace object_detection {

/**
 * \class InterestOperatorFactory
 * \author Stephan Wirth
 * \brief Factory for InterestOperators.
 */
class InterestOperatorFactory
{
public:

    /**
    * Creates and returns a new interest operator by given name. If no
    * operator with the given name is known,
    * a NULL-pointer is returned (i.e. ptr.get() == NULL).
    */
    static boost::shared_ptr<InterestOperator> create(const std::string& name);
    
};

}


#endif /* INTEREST_OPERATOR_FACTORY_H */

