#ifndef DESCRIPTOR_H_
#define DESCRIPTOR_H_


#include <pcl/point_types.h>
#include <pcl/point_representation.h>

namespace object_detection
{
    struct Descriptor
    {
        float data[64];
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    } EIGEN_ALIGN16; 
}

POINT_CLOUD_REGISTER_POINT_STRUCT (object_detection::Descriptor,
                                  (float[64], data, data)
                                  )


namespace pcl
{
template <>
class DefaultPointRepresentation<object_detection::Descriptor> : public PointRepresentation<object_detection::Descriptor>
  {
    public:
      DefaultPointRepresentation ()
      {
        nr_dimensions_ = 64;
      }

      virtual void 
        copyToFloatArray (const object_detection::Descriptor& d, float * out) const
      {
          for (int i = 0; i < nr_dimensions_; ++i)
              out[i] = d.data[i];
      }
  };
}

#endif

