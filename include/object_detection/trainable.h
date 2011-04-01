#ifndef TRAINABLE_H 
#define TRAINABLE_H

namespace object_detection {

struct TrainingData;

/**
 * \class Trainable
 * \author Stephan Wirth
 * \brief Interface for trainable objects.
 */
class Trainable
{
public:

    /**
     * Virtual destructor (empty)
     */
	virtual ~Trainable() {};

    /**
     * \brief training method
     * \param training_data the data that is used for training 
     */
    virtual void train(const TrainingData& training_data) = 0;

    /**
     * \brief checks if the trainable is ready for usage
     * \return true if the trainable is trained and can be used.
     */
    virtual bool isTrained() const = 0;
};

}


#endif /* TRAINABLE_H */

