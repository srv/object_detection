#include "detection.h"

std::ostream& operator<< (std::ostream& out, const object_detection::Detection& detection)
{
    out << "label: " << detection.label << " angle: " << detection.angle
        << " center: (" << detection.center.x << ","
        << detection.center.y << ")" 
        << " scale: " << detection.scale
        << " score: " << detection.score;
    return out;
}

