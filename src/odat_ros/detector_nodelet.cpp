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
**/

#include "odat/detector.h"

#include "odat_ros/detector_nodelet.h"
#include "odat_ros/conversions.h"

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <cv_bridge/cv_bridge.h>

#include <vision_msgs/DetectionArray.h>
#include <geometry_msgs/PoseArray.h>
#include <time.h>
#include <boost/algorithm/string.hpp>

namespace odat_ros {


/**
 * Subscribe to the topics of interest. Depending on
 * some parameters (e.g. use_masks) it can subscribe to different topics.
 */
void DetectorNodelet::subscribeTopics(ros::NodeHandle& nh)
{
	// get subscribe related parameters
	nh.getParam("use_image", use_image_);
	nh.getParam("use_masks", use_masks_);
	nh.getParam("use_input_detections", use_input_detections_);
	nh.getParam("use_point_cloud", use_point_cloud_);
	nh.getParam("queue_size", queue_size_);

	typedef message_filters::Subscriber<vision_msgs::MaskArray> MaskArraySubscriber;
	typedef message_filters::Subscriber<vision_msgs::DetectionArray> DetectionArraySubscriber;
	typedef message_filters::Subscriber<sensor_msgs::PointCloud2> PointCloud2Subscriber;

	// clearing the list unsubscribes from everything subscribed so far
	shared_ptrs_.clear();

	NODELET_INFO("Detector nodelet about to subscribe to:\n"
			     "  image           : %s\n"
			     "  masks           : %s\n"
			     "  input_detections: %s\n"
			     "  point_cloud     : %s\n", (use_image_? nh.resolveName("image").c_str():"None"),
			                        (use_masks_? nh.resolveName("masks").c_str():"None"),
			                        (use_input_detections_? nh.resolveName("input_detections").c_str():"None"),
			                        (use_point_cloud_? nh.resolveName("point_cloud").c_str():"None"));

	if (use_image_) {
		boost::shared_ptr<image_transport::ImageTransport> it = make_shared<image_transport::ImageTransport>(nh);
		boost::shared_ptr<image_transport::SubscriberFilter> image_sub =  make_shared<image_transport::SubscriberFilter>();
		image_sub->subscribe(*it, "image", queue_size_);

		if (use_point_cloud_) {
			boost::shared_ptr<PointCloud2Subscriber> point_cloud_sub =  make_shared<PointCloud2Subscriber>();
			point_cloud_sub->subscribe(nh, "point_cloud", queue_size_);

			if (use_input_detections_) {
				boost::shared_ptr<DetectionArraySubscriber> detections_sub =  make_shared<DetectionArraySubscriber>();
				detections_sub->subscribe(nh, "input_detections", queue_size_);

				typedef message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2, vision_msgs::DetectionArray> SynchronizerImagePointCloudDetections;
				boost::shared_ptr<SynchronizerImagePointCloudDetections> sync_sub = make_shared<SynchronizerImagePointCloudDetections>(queue_size_+2);
				sync_sub->connectInput(*image_sub, *point_cloud_sub, *detections_sub);
				sync_sub->registerCallback(boost::bind(&DetectorNodelet::dataCallback, this, _1, _2, vision_msgs::MaskArrayConstPtr(), _3));

			}
			else {
				if (use_masks_) {
					boost::shared_ptr<MaskArraySubscriber> masks_sub =  make_shared<MaskArraySubscriber>();
					masks_sub->subscribe(nh, "masks", queue_size_);

					typedef message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2, vision_msgs::MaskArray> SynchronizerImagePointCloudMasks;
					boost::shared_ptr<SynchronizerImagePointCloudMasks> sync_sub = make_shared<SynchronizerImagePointCloudMasks>(queue_size_+2);
					sync_sub->connectInput(*image_sub, *point_cloud_sub, *masks_sub);
					sync_sub->registerCallback(boost::bind(&DetectorNodelet::dataCallback, this, _1, _2, _3, vision_msgs::DetectionArrayConstPtr()));
				}
				else {
					typedef message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> SynchronizerImagePointCloud;
					boost::shared_ptr<SynchronizerImagePointCloud> sync_sub = make_shared<SynchronizerImagePointCloud>(queue_size_+2);
					sync_sub->connectInput(*image_sub, *point_cloud_sub);
					sync_sub->registerCallback(boost::bind(&DetectorNodelet::dataCallback, this, _1, _2, vision_msgs::MaskArrayConstPtr(), vision_msgs::DetectionArrayConstPtr()));
				}
			}
		}
		else {
			if (use_input_detections_) {
				boost::shared_ptr<DetectionArraySubscriber> detections_sub =  make_shared<DetectionArraySubscriber>();
				detections_sub->subscribe(nh, "input_detections", queue_size_);

				typedef message_filters::TimeSynchronizer<sensor_msgs::Image, vision_msgs::DetectionArray> SynchronizerImageDetections;
				boost::shared_ptr<SynchronizerImageDetections> sync_sub = make_shared<SynchronizerImageDetections>(queue_size_+2);
				sync_sub->connectInput(*image_sub, *detections_sub);
				sync_sub->registerCallback(boost::bind(&DetectorNodelet::dataCallback, this, _1, sensor_msgs::PointCloud2ConstPtr(), vision_msgs::MaskArrayConstPtr(), _2));
			}
			else {

				if (use_masks_) {
					boost::shared_ptr<MaskArraySubscriber> masks_sub =  make_shared<MaskArraySubscriber>();
					masks_sub->subscribe(nh, "masks", queue_size_);

					typedef message_filters::TimeSynchronizer<sensor_msgs::Image, vision_msgs::MaskArray> SynchronizerImageMaskArray;
					boost::shared_ptr<SynchronizerImageMaskArray> sync_sub = make_shared<SynchronizerImageMaskArray>(queue_size_+2);
					sync_sub->connectInput(*image_sub, *masks_sub);
					sync_sub->registerCallback(boost::bind(&DetectorNodelet::dataCallback, this, _1, sensor_msgs::PointCloud2ConstPtr(), _2, vision_msgs::DetectionArrayConstPtr()));
				}
				else {
					image_sub->registerCallback(boost::bind(&DetectorNodelet::dataCallback, this,_1, sensor_msgs::PointCloud2ConstPtr(), vision_msgs::MaskArrayConstPtr(), vision_msgs::DetectionArrayConstPtr()));
				}
			}
		}
	}
	else {
		if (use_point_cloud_) {
			boost::shared_ptr<PointCloud2Subscriber> point_cloud_sub =  make_shared<PointCloud2Subscriber>();
			point_cloud_sub->subscribe(nh, "point_cloud", queue_size_);

			if (use_input_detections_) {
				boost::shared_ptr<DetectionArraySubscriber> detections_sub =  make_shared<DetectionArraySubscriber>();
				detections_sub->subscribe(nh, "input_detections", queue_size_);

				typedef message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, vision_msgs::DetectionArray> SynchronizerPointCloudDetections;
				boost::shared_ptr<SynchronizerPointCloudDetections> sync_sub = make_shared<SynchronizerPointCloudDetections>(queue_size_+2);
				sync_sub->connectInput(*point_cloud_sub, *detections_sub);
				sync_sub->registerCallback(boost::bind(&DetectorNodelet::dataCallback, this, sensor_msgs::ImageConstPtr(), _1, vision_msgs::MaskArrayConstPtr(), _2));
			}
			else {
				if (use_masks_) {
					boost::shared_ptr<MaskArraySubscriber> masks_sub =  make_shared<MaskArraySubscriber>();
					masks_sub->subscribe(nh, "masks", queue_size_);

					typedef message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, vision_msgs::MaskArray> SynchronizerPointCloudmask;
					boost::shared_ptr<SynchronizerPointCloudmask> sync_sub = make_shared<SynchronizerPointCloudmask>(queue_size_+2);
					sync_sub->connectInput(*point_cloud_sub, *masks_sub);
					sync_sub->registerCallback(boost::bind(&DetectorNodelet::dataCallback, this, sensor_msgs::ImageConstPtr(), _1, _2, vision_msgs::DetectionArrayConstPtr()));
				}
				else {
					point_cloud_sub->registerCallback(boost::bind(&DetectorNodelet::dataCallback, this, sensor_msgs::ImageConstPtr(), _1, vision_msgs::MaskArrayConstPtr(), vision_msgs::DetectionArrayConstPtr()));
				}
			}
		}
	}
}


void DetectorNodelet::advertiseTopics(ros::NodeHandle& nh)
{
	// advertise published topics
	detections_pub_ = nh.advertise<vision_msgs::DetectionArray>("detections",1);
}

/**
 * Checks to see if anybody is interested in the results.
 * @return True is somebody subscribed to results
 */
bool DetectorNodelet::resultsNeeded()
{
	return (detections_pub_.getNumSubscribers()>0);
}

/**
 * Callback
 * @param image the image to run recognition on
 * @param masks the regions of interest in the image (might be missing)
 */
void DetectorNodelet::dataCallback(const sensor_msgs::ImageConstPtr& image_msg,
									const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg,
									const vision_msgs::MaskArrayConstPtr& masks_msg,
									const vision_msgs::DetectionArrayConstPtr& detections_msg)
{
	if (use_image_) {
		header_ = image_msg->header;
	}
	else if (use_point_cloud_){
		header_ = point_cloud_msg->header;
	}

    if (use_image_) 
    {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(image_msg);
      detector_->setImage(cv_ptr->image);
    }

	if (use_masks_)
    {
      std::vector<odat::Mask> masks;
      odat_ros::fromMsg(*masks_msg, masks);
	  detector_->setMasks(masks);
    }

	if (use_input_detections_) 
    {
      std::vector<odat::Detection> detections;
      odat_ros::fromMsg(*detections_msg, detections);
      detector_->setInputdetections(detections);
    }
	//if (use_point_cloud_) detector_->setPointCloud(point_cloud);

	if (resultsNeeded()) {
		runNodelet();
		publishResults();
	}
}


/**
 * Runs the detector.
 */
void DetectorNodelet::runNodelet()
{
	NODELET_DEBUG("Running %s detector", detector_->getName().c_str());
	clock_t start = clock();
	detector_->detect();
	NODELET_DEBUG("Detection took %g seconds.", ((double)clock()-start)/CLOCKS_PER_SEC);
}


/**
 * Publishes detection results
 */
void DetectorNodelet::publishResults()
{
  std::vector<odat::Detection> detections = detector_->getDetections();
  vision_msgs::DetectionArray::Ptr detections_msg = boost::make_shared<vision_msgs::DetectionArray>();
  odat_ros::toMsg(detections, *detections_msg);
  detections_msg->header = header_;
  detections_pub_.publish(detections_msg);
}


/**
 * Loads a list of models into the detector
 * @param models  comma-separated list of models
 */
void DetectorNodelet::loadModels(const std::string& models)
{
	if (models=="__none__") return;
	std::vector<std::string> models_vec;
	boost::split(models_vec, models, boost::is_any_of("\t ,"));
	std::vector<std::string> models_vec_filtered;
	for (size_t i=0;i<models_vec.size();++i) {
		if (!models_vec[i].empty()) models_vec_filtered.push_back(models_vec[i]);
	}

	if (models_vec_filtered.size()!=0) {
		detector_->loadModels(models_vec_filtered);
	}
	else {
		detector_->loadAllModels();
	}
}

}
