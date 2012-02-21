#ifndef PCL_DESCRIPTOR_H_
#define PCL_DESCRIPTOR_H_


#include <pcl/point_types.h>
#include <pcl/point_representation.h>

namespace object_detection
{

struct PclDescriptor
{
  float data[64];
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

inline std::ostream& operator<<(std::ostream& ostr, const PclDescriptor& desc)
{
  ostr << "(" << desc.data[0];
  for (int i = 1; i < 64; ++i)
  {
    ostr << ", " << desc.data[i];
  }
  ostr << ")";
  return ostr;
}

}

POINT_CLOUD_REGISTER_POINT_STRUCT (object_detection::PclDescriptor,
                                   (float[64], data, data)
                                   )


namespace pcl
{
template <>
class DefaultPointRepresentation<object_detection::PclDescriptor> : 
  public PointRepresentation<object_detection::PclDescriptor>
{
public:
  DefaultPointRepresentation ()
  {
    nr_dimensions_ = 64;
  }

  virtual void
  copyToFloatArray (const object_detection::PclDescriptor& d, float * out) const
  {
    for (int i = 0; i < nr_dimensions_; ++i)
      out[i] = d.data[i];
  }
};
}

#endif

