#ifndef MASK_H_
#define MASK_H_

#include <opencv2/core/core.hpp>

namespace odat
{
  struct Mask
  {
    // location of the mask that has same size as mask
    cv::Rect roi;

    // if set, binary pattern
    cv::Mat mask; 
  };
}

#endif
