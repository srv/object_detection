#ifndef MASK_H
#define MASK_H

#include <cv.h>

namespace object_detection {

/**
 * \struct Mask
 * \author Stephan Wirth
 * \brief Data structure that defines an arbitrary mask for an image
 * A mask is defined as a rectangle and an image of the same size.
 * The rectangle defines the position of the (binary) image mask in a
 * target image.
 */
struct Mask 
{
    /// the rectangle that defines the position of the image mask
    cv::Rect rect;

    /// the image mask that masks an object of interest.
    /// a value above zero means that the pixel belongs to the object.
    /// As a convention if the image has zero size, the whole rect
    /// masks the object (same as if all values in the image are
    /// above zero).
    cv::Mat image;
};

}

#endif /* MASK_H */

