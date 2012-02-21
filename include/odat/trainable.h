#ifndef TRAINABLE_H_
#define TRAINABLE_H_

#include <string>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

#include "odat/training_data.h"

namespace odat {

  /**
  * Interface for an algorithm capable of training models
  */
  class Trainable
  {
    public:
      /**
      * Starts training for a new object model. It may allocate/initialize
      * data structures needed for training a new category.
      * @param name
      */
      virtual void startTraining(const std::string& name) = 0;

      /**
      * Trains the model on a new data instance.
      * @param name The name of the model
      * @param data Training data instance
      */
      virtual void trainInstance(const std::string& name, const odat::TrainingData& data) = 0;

      /**
      * Finalizes the training for given model name and saves it.
      * @param name model name
      */
      virtual void endTraining(const std::string& name) = 0;

      typedef boost::shared_ptr<Trainable> Ptr;
      typedef boost::shared_ptr<const Trainable> ConstPtr;
  };
}

#endif /* TRAINABLE_H_ */

