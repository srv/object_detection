/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

/**
\author Marius Muja
\author Stephan Wirth
 **/

#include <sstream>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <vision_msgs/DetectionArray.h>

#include "odat_ros/conversions.h"

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
        image_sub_.subscribe(it_,"image", 1);
        detections_sub_.subscribe(nh_,"detections", 1);

        sync_sub_.connectInput(image_sub_,detections_sub_);
        sync_sub_.registerCallback(&DetectionDisplay::detectionCallback, this);

        nh_.param<std::string>("window_name", window_name_, "Detections");
        cv::namedWindow(window_name_, 0);
      }


	void detectionCallback(const sensor_msgs::ImageConstPtr& img_msg,
			const vision_msgs::DetectionArrayConstPtr& detections_msg)
	{
      cv::Mat image = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
 
      int image_width = image.cols;
      int image_height = image.rows;

      std::vector<odat::Detection> detections;
      odat_ros::fromMsg(*detections_msg, detections);

      for (size_t i = 0; i < detections.size(); ++i) 
      {
        int x = detections[i].mask.roi.x;
        int y = detections[i].mask.roi.y;
        int w = detections[i].mask.roi.width;
        int h = detections[i].mask.roi.height;

        int bl = (2*2555)%200;
        int gr = (2*433)%224;
        int rd = (2*2020)%210;
        cv::rectangle(image, detections[i].mask.roi.tl(), detections[i].mask.roi.br(), cv::Scalar(bl,gr,rd), 2);

        cv::Mat mask = detections[i].mask.mask;
        if (mask.data != NULL) {
          int source_x = x < 0 ? -x : 0;
          int source_y = y < 0 ? -y : 0;
          int dest_x = x < 0 ? 0 : x;
          int dest_y = y < 0 ? 0 : y;
          int width = w;
          int height = h;
          if (x < 0) width += x;
          if (y < 0) height += y;
          if (dest_x + width >= image_width) width = image_width - dest_x - 1;
          if (dest_y + height >= image_height) height = image_height - dest_y - 1;

          cv::Rect copy_roi(source_x, source_y, width, height);
          cv::Mat source = mask(copy_roi);
          copy_roi.x = dest_x;
          copy_roi.y = dest_y;
          cv::Mat dest = image(copy_roi);

          cv::add(dest, cv::Scalar(0, 80, 0), dest, source);
        }

        int baseline = 0;
        int X = x+3, Y = y+12;
        cv::Size tsize = getTextSize(detections[i].label, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        if( Y + tsize.height + 2 >= image_height) { Y = image_height - y - 12 - 2 - tsize.height;}
        if(Y < 0) Y = 0;
        if(X + tsize.width + 2 >= image_width) { X = image_width - x - 3 - 2 - tsize.width;}
        if(X < 0) X = 0;
        cv::putText(image, detections[i].label,cv::Point(X,Y),cv::FONT_HERSHEY_SIMPLEX,1.0,cv::Scalar(bl,gr,rd),2);
        std::string strscore = std::string("(") + tostr(detections[i].score) + std::string(")");
        X = x + 6; Y = y+24;
        tsize = getTextSize(strscore,cv::FONT_HERSHEY_SIMPLEX,0.4,1,&baseline);
        if( Y + tsize.height + 1 >= image_height) { Y = image_height - y - 24 - 1 - tsize.height;}
        if(Y < 0) Y = 0;
        if(X + tsize.width + 1 >= image_width) { X = image_width - x - 3 - 1 - tsize.width;}
        if(X < 0) X = 0;
        putText(image,strscore,cv::Point(X,Y),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(bl,gr,rd),1);
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

