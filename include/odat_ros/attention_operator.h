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

#ifndef ATTENTION_NODELET_H_
#define ATTENTION_NODELET_H_

#include "odat/nodelet_base.h"
#include "odat/roi.h"


namespace odat_ros {

/**
 * This class implements the interface for an attention operator nodelet that is capable
 * of being dynamically loaded/unloaded, and it automatically subscribes to the
 * corresponding topics depending on the parameters with which is configured.
 *
 * For a working attention operator this class must be extended and the \c runAttentionOperator()
 * method must be implemented in the subclass. This method can make use of the fields
 * \c image_ and \c point_cloud_ which are automatically set by the
 * framework and store the results in the \c rois_ field.
 *
 * The following parameters influence the topics to which the nodelet subscribes:
 *  \li \c use_image_ Makes the nodelet subscribe to the image topic
 *  \li \c use_point_cloud  Makes the nodelet subscribe to the point cloud topic
 *
 *  All the topic to which the nodelet subscribes are in its private namespace and
 *  when it subscribes to two or more topics, they are synchronized by timestamp.
 *
 */

class AttentionOperator;


class AttentionOperatorNodelet : public NodeletBase
{
public:
	AttentionOperatorNodelet() : use_image_(true), use_point_cloud_(false), approximate_sync_(false), queue_size_(10) {};

protected:
	/**
	 * Subscribe to the topics of interest. Depending on
	 * some parameters (e.g. use_rois) it can subscribe to different topics.
	 */
	virtual void subscribeTopics(ros::NodeHandle& nh);

	/**
	 *
	 * @param nh
	 */
	virtual void advertiseTopics(ros::NodeHandle& nh);

	/**
	 * Checks to see if anybody is interested in the results.
	 * @return True is somebody subscribed to results
	 */
	virtual bool resultsNeeded();

	/**
	 * Runs the attention operator
	 */
	virtual void runNodelet();

	/**
	 * Publishes results
	 */
	virtual void publishResults();

private:
	/**
	 * Callback
	 * @param image the image to run recognition on
	 * @param rois the regions of interest in the image (might be missing)
	 */
	void dataCallback(const sensor_msgs::ImageConstPtr& image,
			const sensor_msgs::PointCloud2ConstPtr& point_cloud);

protected:
	bool use_image_;
	bool use_point_cloud_;
	bool approximate_sync_;
	int queue_size_;

	boost::shared_ptr<rein::AttentionOperator> attention_operator_;

private:
    // we could use a header message here directly, but we are using
    // Image (actually just the header from Image) for
    // the code to be portable with the move of the Header message
    // from roslib to std_msgs
    sensor_msgs::Image header_;
    //roslib::Header header_;

	// publishers
	ros::Publisher rois_pub_;
	ros::Publisher masks_pub_;
};

}

#endif /* ATTENTION_NODELET_H_ */
