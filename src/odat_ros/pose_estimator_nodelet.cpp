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

#include "rein/nodelets/pose_estimator.h"
#include "rein/core/pose_estimator.h"

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <cv_bridge/CvBridge.h>

namespace rein {


/**
 * Subscribe to the topics of interest. Depending on
 * some parameters (e.g. use_rois) it can subscribe to different topics.
 */
void PoseEstimatorNodelet::subscribeTopics(ros::NodeHandle& nh)
{
	nh.getParam("use_image", use_image_);
	nh.getParam("use_detections", use_detections_);
	nh.getParam("use_point_cloud", use_point_cloud_);
	nh.getParam("queue_size", queue_size_);

	typedef message_filters::Subscriber<rein::DetectionArray> DetectionArraySubscriber;
	typedef message_filters::Subscriber<sensor_msgs::PointCloud2> PointCloud2Subscriber;

	// clearing the list unsubscribes from everything subscribed so far
	shared_ptrs_.clear();

	if (use_image_) {
		boost::shared_ptr<image_transport::ImageTransport> it = make_shared<image_transport::ImageTransport>(nh);
		boost::shared_ptr<image_transport::SubscriberFilter> image_sub =  make_shared<image_transport::SubscriberFilter>();
		image_sub->subscribe(*it, "image", 1);

		if (use_point_cloud_) {
			boost::shared_ptr<PointCloud2Subscriber> point_cloud_sub =  make_shared<PointCloud2Subscriber>();
			point_cloud_sub->subscribe(nh, "point_cloud", 1);

			if (use_detections_) {
				boost::shared_ptr<DetectionArraySubscriber> detections_sub =  make_shared<DetectionArraySubscriber>();
				detections_sub->subscribe(nh, "detections", 1);

				typedef message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2, rein::DetectionArray> SynchronizerImagePointCloudDetections;
				boost::shared_ptr<SynchronizerImagePointCloudDetections> sync_sub = make_shared<SynchronizerImagePointCloudDetections>(3);
				sync_sub->connectInput(*image_sub, *point_cloud_sub, *detections_sub);
				sync_sub->registerCallback(boost::bind(&PoseEstimatorNodelet::dataCallback, this, _1, _2, _3));
			}
			else {
				// probably not really useful without detections, but here for completeness
				typedef message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> SynchronizerImagePointCloud;
				boost::shared_ptr<SynchronizerImagePointCloud> sync_sub = make_shared<SynchronizerImagePointCloud>(3);
				sync_sub->connectInput(*image_sub, *point_cloud_sub);
				sync_sub->registerCallback(boost::bind(&PoseEstimatorNodelet::dataCallback, this, _1, _2, DetectionArrayConstPtr()));
			}
		}
		else {
			if (use_detections_) {
				boost::shared_ptr<DetectionArraySubscriber> detections_sub =  make_shared<DetectionArraySubscriber>();
				detections_sub->subscribe(nh, "detections", 1);

				typedef message_filters::TimeSynchronizer<sensor_msgs::Image, rein::DetectionArray> SynchronizerImageDetections;
				boost::shared_ptr<SynchronizerImageDetections> sync_sub = make_shared<SynchronizerImageDetections>(3);
				sync_sub->connectInput(*image_sub, *detections_sub);
				sync_sub->registerCallback(boost::bind(&PoseEstimatorNodelet::dataCallback, this, _1, sensor_msgs::PointCloud2ConstPtr(), _2));
			}
			else {
				// probably not really useful without detections, but here for completeness
				image_sub->registerCallback(boost::bind(&PoseEstimatorNodelet::dataCallback, this,_1, sensor_msgs::PointCloud2ConstPtr(), DetectionArrayConstPtr()));
			}
		}
	}
	else {
		if (use_point_cloud_) {
			boost::shared_ptr<PointCloud2Subscriber> point_cloud_sub =  make_shared<PointCloud2Subscriber>();
			point_cloud_sub->subscribe(nh, "point_cloud", 1);

			if (use_detections_) {
				boost::shared_ptr<DetectionArraySubscriber> detections_sub =  make_shared<DetectionArraySubscriber>();
				detections_sub->subscribe(nh, "detections", 1);

				typedef message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, rein::DetectionArray> SynchronizerPointCloudDetections;
				boost::shared_ptr<SynchronizerPointCloudDetections> sync_sub = make_shared<SynchronizerPointCloudDetections>(3);
				sync_sub->connectInput(*point_cloud_sub, *detections_sub);
				sync_sub->registerCallback(boost::bind(&PoseEstimatorNodelet::dataCallback, this, sensor_msgs::ImageConstPtr(), _1, _2));
			}
			else {
				point_cloud_sub->registerCallback(boost::bind(&PoseEstimatorNodelet::dataCallback, this, sensor_msgs::ImageConstPtr(), _1, DetectionArrayConstPtr()));
			}
		}
	}

}


void PoseEstimatorNodelet::advertiseTopics(ros::NodeHandle& nh)
{
	poses_pub_ = nh.advertise<geometry_msgs::PoseArray>("poses",1);
}


bool PoseEstimatorNodelet::resultsNeeded()
{
	return poses_pub_.getNumSubscribers()>0;
}

/**
 * Callback
 * @param image the image to run recognition on
 * @param detections array of detected objects in the input image
 */
void PoseEstimatorNodelet::dataCallback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::PointCloud2ConstPtr& point_cloud, const rein::DetectionArrayConstPtr& detections)
{
	static sensor_msgs::CvBridge cv_bridge;

	if (use_image_) pose_estimator_->setImage(cv_bridge.imgMsgToCv(image));
	if (use_point_cloud_) pose_estimator_->setPointCloud(point_cloud);
	if (use_detections_) pose_estimator_->setDetections(detections);

	if (use_image_) {
		header_.header = image->header;
	}
	else if (use_point_cloud_){
		header_.header = point_cloud->header;
	}

	if (resultsNeeded()) {
		runNodelet();
		publishResults();
	}
}


void PoseEstimatorNodelet::runNodelet()
{
	pose_estimator_->run();
}

/**
 * Publishes detection results
 */
void PoseEstimatorNodelet::publishResults()
{
	geometry_msgs::PoseArrayPtr poses = boost::make_shared<geometry_msgs::PoseArray>(pose_estimator_->getPoses());

	// publish detections
	poses->header = header_.header;
	poses_pub_.publish(poses);
}


}
