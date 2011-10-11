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

#include "rein/nodelets/attention_operator.h"
#include "rein/core/attention_operator.h"
#include "rein/RectArray.h"
#include <cv_bridge/CvBridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>



namespace rein {

/**
 * Subscribe to the topics of interest. Depending on
 * some parameters (e.g. use_rois) it can subscribe to different topics.
 */
void AttentionOperatorNodelet::subscribeTopics(ros::NodeHandle& nh)
{
	nh.getParam("use_image", use_image_);
	nh.getParam("use_point_cloud", use_point_cloud_);
	nh.getParam("approximate_sync", approximate_sync_);
	nh.getParam("queue_size", queue_size_);

	typedef message_filters::Subscriber<rein::RectArray> RectArraySubscriber;
	typedef message_filters::Subscriber<sensor_msgs::PointCloud2> PointCloud2Subscriber;

	// clearing the list unsubscribes from everything subscribed so far
	shared_ptrs_.clear();

	if (use_image_) {
		boost::shared_ptr<image_transport::ImageTransport> it = make_shared<image_transport::ImageTransport>(nh);
		boost::shared_ptr<image_transport::SubscriberFilter> image_sub =  make_shared<image_transport::SubscriberFilter>();
		image_sub->subscribe(*it, "image", queue_size_);

		if (use_point_cloud_) {
			boost::shared_ptr<PointCloud2Subscriber> point_cloud_sub =  make_shared<PointCloud2Subscriber>();
			point_cloud_sub->subscribe(nh, "point_cloud", queue_size_);

			if (approximate_sync_) {
				typedef message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> > SynchronizerImagePointCloud;
				boost::shared_ptr<SynchronizerImagePointCloud> sync_sub = make_shared<SynchronizerImagePointCloud>(queue_size_+2);
				sync_sub->connectInput(*image_sub, *point_cloud_sub);
				sync_sub->registerCallback(boost::bind(&AttentionOperatorNodelet::dataCallback, this, _1, _2));
			}
			else {
				typedef message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> SynchronizerImagePointCloud;
				boost::shared_ptr<SynchronizerImagePointCloud> sync_sub = make_shared<SynchronizerImagePointCloud>(queue_size_+2);
				sync_sub->connectInput(*image_sub, *point_cloud_sub);
				sync_sub->registerCallback(boost::bind(&AttentionOperatorNodelet::dataCallback, this, _1, _2));
			}
		}
		else {
			image_sub->registerCallback(boost::bind(&AttentionOperatorNodelet::dataCallback, this,_1, sensor_msgs::PointCloud2ConstPtr()));
		}
	}
	else {
		if (use_point_cloud_) {
			boost::shared_ptr<PointCloud2Subscriber> point_cloud_sub =  make_shared<PointCloud2Subscriber>();
			point_cloud_sub->subscribe(nh, "point_cloud", queue_size_);

			point_cloud_sub->registerCallback(boost::bind(&AttentionOperatorNodelet::dataCallback, this, sensor_msgs::ImageConstPtr(), _1));
		}
	}
}

/**
 *
 */
void AttentionOperatorNodelet::advertiseTopics(ros::NodeHandle& nh)
{
	masks_pub_ = nh.advertise<MaskArray>("masks",1);
	rois_pub_ = nh.advertise<RectArray>("rois",1);
}

/**
 *
 * @return
 */
bool AttentionOperatorNodelet::resultsNeeded()
{
	return ((rois_pub_.getNumSubscribers()>0) || (masks_pub_.getNumSubscribers()>0));
}


/**
 *
 */
void AttentionOperatorNodelet::runNodelet()
{
	attention_operator_->run();
}

/**
 * Callback
 * @param image the image to run recognition on
 * @param rois the regions of interest in the image (might be missing)
 */
void AttentionOperatorNodelet::dataCallback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::PointCloud2ConstPtr& point_cloud)
{
	static sensor_msgs::CvBridge cv_bridge;

	if (use_image_) {
		header_.header = image->header;
	}
	else if (use_point_cloud_){
		header_.header = point_cloud->header;
	}

	if (use_image_) attention_operator_->setImage(cv_bridge.imgMsgToCv(image));
	if (use_point_cloud_) attention_operator_->setPointCloud(point_cloud);

	if (resultsNeeded()) {
		runNodelet();
		publishResults();
	}
}


/**
 * Publish attention operator results
 */
void AttentionOperatorNodelet::publishResults()
{
	// publish detections
	MaskArrayPtr masks = boost::make_shared<MaskArray>(attention_operator_->getMasks());
	masks->header = header_.header;
	masks_pub_.publish(masks);

	if (rois_pub_.getNumSubscribers()>0) {
		// publish bounding boxes
		RectArrayPtr rois = boost::make_shared<RectArray>();
		rois->rects.resize(masks->masks.size());
		for (size_t i=0;i<masks->masks.size();++i) {
			rois->rects[i] = masks->masks[i].roi;
		}
		rois->header = header_.header;
		rois_pub_.publish(rois);
	}
}


}
