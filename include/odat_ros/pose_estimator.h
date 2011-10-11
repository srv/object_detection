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

#ifndef POSE_ESTIMATOR_NODELET_H_
#define POSE_ESTIMATOR_NODELET_H_

#include "rein/nodelets/nodelet_base.h"

#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseArray.h>
#include <rein/DetectionArray.h>
#include <sensor_msgs/PointCloud2.h>


namespace odat_ros {


class PoseEstimator;

/**
 * This class implements the interface for a pose estimation nodelet that is capable
 * of being dynamically loaded/unloaded, and it automatically subscribes to the
 * corresponding topics depending on the parameters with which is configured.
 *
 * For a working detector this class must be extended and the \c runPoseEstimation()
 * method must be implemented in the subclass. This method can make use of the fields
 * \c image_, \c point_cloud_ and \c detections_ which are automatically set by the
 * framework and store the pose estimation results in the \c poses_ field.
 *
 * The following parameters influence the topics to which the nodelet subscribes:
 *  \li \c use_image_ Makes the nodelet subscribe to the image topic
 *  \li \c use_point_cloud_  Makes the nodelet subscribe to the point cloud topic
 *  \li \c use_detections_ Makes the nodelet subscribe to the detections topic
 *
 *  All the topic to which the nodelet subscribes are in its private namespace and
 *  when it subscribes to two or more topics, they are synchronized by timestamp.
 *
 */
class PoseEstimatorNodelet : public NodeletBase
{
public:
	PoseEstimatorNodelet() : use_image_(false), use_point_cloud_(true), use_detections_(true), queue_size_(10) {};

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
	 * Runs the pose estimation.
	 */
	virtual void runNodelet();

	/**
	 * Publishes detection results
	 */
	virtual void publishResults();

private:
	/**
	 * Callback
	 * @param image the image to run recognition on
	 * @param rois the regions of interest in the image (might be missing)
	 */
	void dataCallback(const sensor_msgs::ImageConstPtr& image,
			const sensor_msgs::PointCloud2ConstPtr& point_cloud,
			const rein::DetectionArrayConstPtr& detections);

protected:
    // we could use a header message here directly, but we are using
    // Image (actually just the header from Image) for
    // the code to be portable with the move of the Header message
    // from roslib to std_msgs
    sensor_msgs::Image header_;
    //roslib::Header header_;

	bool use_image_;
	bool use_point_cloud_;
	bool use_detections_;
	int queue_size_;

private:
	// publishers
	ros::Publisher poses_pub_;

	boost::shared_ptr<PoseEstimator> pose_estimator_;
};

}


#endif /* POSE_ESTIMATOR_NODELET_H_ */
