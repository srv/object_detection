#include "interest_operator_factory.h"
#include "histogram_backprojection.h"

using boost::shared_ptr;
using object_detection::InterestOperator;
using object_detection::InterestOperatorFactory;

shared_ptr<InterestOperator> InterestOperatorFactory::create(const std::string& name)
{
    if (name == "HistogramBackprojection")
    {
        shared_ptr<InterestOperator> op(new HistogramBackprojection());
        return op;
    }
    else
    {
        shared_ptr<InterestOperator> op;
        return op;
    }
}

