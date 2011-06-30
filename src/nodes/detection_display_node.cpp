#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#if ROS_VERSION_MINIMUM(1,4,5)
    #include <cv_bridge/cv_bridge.h>
#else
    #include <cv_bridge/CvBridge.h>
#endif

#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>

#include "object_detection/utilities.h"
#include "vision_msgs/DetectionStamped.h"

namespace enc = sensor_msgs::image_encodings;



class DetectionDisplayNode
{
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    image_transport::ImageTransport it_;
    image_transport::SubscriberFilter image_sub_;

    message_filters::Subscriber<vision_msgs::DetectionStamped> detection_sub_;
    
    message_filters::TimeSynchronizer<sensor_msgs::Image, vision_msgs::DetectionStamped> synchronizer_;

#if ROS_VERSION_MINIMUM(1,4,5)
#else
    sensor_msgs::CvBridge bridge_;
#endif

    bool paint_rectangle_;
    bool paint_direction_;
    bool paint_outline_;
    bool paint_text_;

public:
    DetectionDisplayNode() :
        nh_private_("~"),
        it_(nh_),
        image_sub_(it_, "image", 1),
        detection_sub_(nh_, "detection", 1),
        // we have to wait for some images here because detection might be slow
        synchronizer_(image_sub_, detection_sub_, 150)

    {
        nh_private_.param<bool>("paint_rectangle", paint_rectangle_, true);
        nh_private_.param<bool>("paint_direction", paint_direction_, true);
        nh_private_.param<bool>("paint_outline", paint_outline_, true);
        nh_private_.param<bool>("paint_text", paint_text_, true);
        synchronizer_.registerCallback(boost::bind(&DetectionDisplayNode::detectionCallback, this, _1, _2));
    }

    ~DetectionDisplayNode()
    {
    }

    void detectionCallback(const sensor_msgs::ImageConstPtr& image_msg,
            const vision_msgs::DetectionStampedConstPtr& detection_msg)
    {

        cv::Mat image_with_detection;

#if ROS_VERSION_MINIMUM(1,4,5)
        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(image_msg, enc::BGR8);
            image_with_detection = cv_ptr->image.clone();
        }
        catch (cv_bridge::Exception& e)
        {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
        }
#else
        try
        {
            IplImage *cv_image = NULL;
            cv_image = bridge_.imgMsgToCv(image_msg, "bgr8");
            image_with_detection = cv::cvarrToMat(cv_image).clone();
        }
        catch (sensor_msgs::CvBridgeException& e)
        {
            ROS_ERROR("CvBridgeException: %s", e.what());
            return;
        }
#endif

        paintDetection(image_with_detection, detection_msg->detection);

        cv::namedWindow("Detection Display");
        cv::imshow("Detection Display", image_with_detection);
        cv::waitKey(3);
    }

    void paintDetection(cv::Mat& image, const vision_msgs::Detection& detection)
    {
        cv::Point center(detection.pose2D.x, detection.pose2D.y);
        if (paint_rectangle_)
        {
            cv::RotatedRect rect(center, 
                    cv::Size(50 * detection.scale, 50 * detection.scale), 
                    detection.pose2D.theta / M_PI * 180.0);
            object_detection::paintRotatedRectangle(image, rect, cv::Scalar(0, 255, 0), 2);
        }

        if (paint_direction_)
        {
            double radius = 20;
            cv::Point direction_point(radius * cos(detection.pose2D.theta), 
                    radius * sin(detection.pose2D.theta));
            cv::line(image, center, center + direction_point, cv::Scalar(0, 255, 0), 3);
            cv::line(image, center, center + direction_point, cv::Scalar(0, 0, 255), 2);
        }

        if (paint_outline_)
        {
            for (size_t i = 0; i < detection.outline.points.size(); ++i)
            {
                cv::Point point1;
                point1.x = detection.outline.points[i].x;
                point1.y = detection.outline.points[i].y;
                cv::Point point2;
                point2.x = detection.outline.points[(i+1) % detection.outline.points.size()].x;
                point2.y = detection.outline.points[(i+1) % detection.outline.points.size()].y;
                cv::line(image, point1, point2, cv::Scalar(0, 0, 255), 2);
            }
        }

        if (paint_text_)
        {
            static const int FONT = CV_FONT_HERSHEY_PLAIN;
            cv::putText(image, detection.id, center, 
                    FONT, 0.8, cv::Scalar(0, 255, 0));
            std::ostringstream ostr;
            ostr << "scale: " << detection.scale;
            cv::putText(image, ostr.str(), center + cv::Point(5, 10), 
                    FONT, 0.8, cv::Scalar(0, 255, 0));
            ostr.str("");
            ostr << "score: " << detection.score;
            cv::putText(image, ostr.str(), center + cv::Point(5, 20), 
                    FONT, 0.8, cv::Scalar(0, 255, 0));
            ostr.str("");
            ostr << "angle: " << detection.pose2D.theta / M_PI * 180.0;
            cv::putText(image, ostr.str(), center + cv::Point(5, 30), 
                    FONT, 0.8, cv::Scalar(0, 255, 0));
        }
    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "detection_display");
  DetectionDisplayNode display;
  ros::spin();
  return 0;
}

