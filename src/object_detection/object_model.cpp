
//#include <GL/gl.h>

#include "object_model.h"

using namespace object_detection;

ObjectModel::ObjectModel()
{
}

void ObjectModel::addFeature3D(const cv::Point3d& world_point,
        const std::vector<Feature>& features, const cv::Scalar& color)
{
    Feature3D feature3d;
    feature3d.world_point = world_point;
    feature3d.features = features;
    feature3d.color = color;
    features_3d_.push_back(feature3d);
}

/*
void ObjectModel::renderGL()
{
    glBegin(GL_POINTS);
    for (size_t i = 0; i < features_3d_.size(); ++i)
    {
        const Feature3D& feature = features_3d_[i];
        glColor3f(feature.color[0], feature.color[1], feature.color[2]);
        glVertex3f(feature.world_point.x, 
                  feature.world_point.y, 
                  feature.world_point.z);
    }
    glEnd();
}

*/
