#include <sstream>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/image_encodings.h>
#include <vision_msgs/DetectionArray.h>

namespace enc = sensor_msgs::image_encodings;

template <typename T>
std::string tostr(const T& t)
{
	std::ostringstream os;
	os << t;
	return os.str();
}

namespace odat_ros
{

  class DetectionDisplay 
  {

    public:

      DetectionDisplay() : it_(nh_), sync_sub_(20)
      {
        image_sub_.subscribe(it_,"image", 5);
        detections_sub_.subscribe(nh_,"detections", 5);

        sync_sub_.connectInput(image_sub_,detections_sub_);
        sync_sub_.registerCallback(&DetectionDisplay::detectionCallback, this);

        nh_.param<std::string>("window_name", window_name_, "Detections");
        cv::namedWindow(window_name_, 0);
      }


	void detectionCallback(const sensor_msgs::ImageConstPtr& image_msg,
			const vision_msgs::DetectionArrayConstPtr& detections_msg)
	{
      cv::Mat image;
      cv_bridge::CvImageConstPtr cv_ptr;
      try
      {
          cv_ptr = cv_bridge::toCvShare(image_msg, enc::BGR8);
          image = cv_ptr->image.clone();
      }
      catch (cv_bridge::Exception& e)
      {
          ROS_ERROR("cv_bridge exception: %s", e.what());
          return;
      }

      for (size_t i = 0; i < detections_msg->detections.size(); ++i) 
      {
        const vision_msgs::Detection& detection = detections_msg->detections[i];
        int x = detection.mask.roi.x;
        int y = detection.mask.roi.y;
        int w = detection.mask.roi.width;
        int h = detection.mask.roi.height;

        cv::Scalar color(0, 255, 170);
        cv::Point top_left(x, y);
        cv::Point bottom_right = top_left + cv::Point(w, h);
        cv::rectangle(image, top_left, bottom_right, color, 2);

        static const int FONT = CV_FONT_HERSHEY_SIMPLEX;
        static const double TEXT_SCALE = 0.8;
        int baseline;
        cv::Size text_size = cv::getTextSize(detection.object_id, FONT, TEXT_SCALE, 2, &baseline);
        cv::putText(image, detection.object_id, cv::Point(x + w + 10, y + 10 + text_size.height), FONT, TEXT_SCALE, color, 2);
        std::string strscore = std::string("score: ") + tostr(detection.score);
        cv::putText(image, strscore, cv::Point(x + w + 10, y + 10 + 2 * text_size.height), FONT, 0.6, color, 2);

        // coordinate system
        cv::Point origin;
        origin.x = detection.image_pose.x;
        origin.y = detection.image_pose.y;
        double direction = detection.image_pose.theta;
        cv::Point x_axis(20 * detection.scale * cos(direction), 20 * detection.scale * sin(direction));
        cv::Point y_axis(20 * detection.scale * cos(direction + M_PI_2), 20 * detection.scale * sin(direction + M_PI_2));
        cv::line(image, origin, origin + x_axis, cv::Scalar(0, 0, 255), 2);
        cv::line(image, origin, origin + y_axis, cv::Scalar(0, 255, 0), 2);
      }

      cv::imshow(window_name_, image);
      cv::waitKey(40);
	}


private:
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;

	image_transport::SubscriberFilter image_sub_;
	message_filters::Subscriber<vision_msgs::DetectionArray> detections_sub_;
	message_filters::TimeSynchronizer<sensor_msgs::Image, vision_msgs::DetectionArray> sync_sub_;

    std::string window_name_;

};

}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "detection_display");
    odat_ros::DetectionDisplay dd;
	ros::spin();
}

