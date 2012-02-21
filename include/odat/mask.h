#ifndef MASK_H_
#define MASK_H_

#include <opencv2/core/core.hpp>

namespace odat
{

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
  /// location of the mask that has same size as mask
  cv::Rect roi;

  /// the image mask that masks an object of interest.
  /// a value above zero means that the pixel belongs to the object.
  /// As a convention if the image has zero size, the whole rect
  /// masks the object (same as if all values in the image are
  /// above zero).
  cv::Mat mask;
};

}

#endif
