#include "odat/detection.h"

std::ostream& operator<< (std::ostream& out, const odat::Detection& detection)
{
  out << "Mask          : \n"
         "    Roi       : (" << detection.mask.roi.x << "," << detection.mask.roi.y << ") (" << detection.mask.roi.width << "x" << detection.mask.roi.height << ")\n"
         "    Mask      : " << (detection.mask.mask.data == NULL ? "empty" : "filled") << "\n"
         "Label         : " << detection.object_id << "\n"
         "Detector      : " << detection.detector << "\n"
         "Score         : " << detection.score << "\n"
         "Image Pose    : (" << detection.image_pose.x << ", " << detection.image_pose.y << ", " << detection.image_pose.theta << ")\n"
         "Scale         : " << detection.scale;
  return out;
}
