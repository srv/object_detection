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

#ifndef DETECTOR_NODELET_H_
#define DETECTOR_NODELET_H_

// messages
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <vision_msgs/DetectionArray.h>
#include <vision_msgs/MaskArray.h>

#include "odat/detector.h"
#include "odat_ros/nodelet_base.h"

namespace odat
{
  class Detector;
}

namespace odat_ros {
/**
 * This class implements the interface for a detector nodelet that is capable
 * of being dynamically loaded/unloaded, and it automatically subscribes to the
 * corresponding topics depending on the parameters with which is configured.
 *
 * For a working detector this class must be extended and the \c runDetector()
 * method must be implemented in the subclass. This method can make use of the fields
 * \c image_, \c point_cloud_ and \c masks_ which are automatically set by the
 * framework and store the detection results in the \c detections_ field.
 *
 * The following parameters influence the topics to which the nodelet subscribes:
 *  \li \c use_image_ Makes the nodelet subscribe to the image topic
 *  \li \c use_point_cloud_  Makes the nodelet subscribe to the point cloud topic
 *  \li \c use_masks_ Makes the nodelet subscribe to the rois topic
 *
 *  All the topic to which the nodelet subscribes are in its private namespace and
 *  when it subscribes to two or more topics, they are synchronized by timestamp.
 *
 */
class DetectorNodelet : public NodeletBase
{
public:
	DetectorNodelet() : use_image_(true), use_point_cloud_(false), use_masks_(false), use_input_detections_(false), queue_size_(1) {};

protected:
	/**
	 * Subscribe to the topics of interest. Depending on
	 * some parameters (e.g. use_masks) it can subscribe to different topics.
	 */
	virtual void subscribeTopics(ros::NodeHandle& nh);

	/**
	 * Advertise the topics this nodelet publishes.
	 */
	virtual void advertiseTopics(ros::NodeHandle& nh);

	/**
	 * Checks to see if anybody is interested in the results.
	 * @return True is somebody subscribed to results
	 */
	virtual bool resultsNeeded();

	/**
	 * Runs the object detection.
	 */
	virtual void runNodelet();

	/**
	 * Publishes detection results
	 */
	virtual void publishResults();

protected:
	/**
	 * Loads a list of models into the detector
	 * @param models  comma-separated list of models
	 */
	void loadModels(const std::string& models);

private:
	/**
	 * Callback
	 * @param image the image to run recognition on
	 * @param masks the regions of interest in the image (might be missing)
	 */
	void dataCallback(const sensor_msgs::ImageConstPtr& image,
			const sensor_msgs::PointCloud2ConstPtr& point_cloud,
			const vision_msgs::MaskArrayConstPtr& masks,
			const vision_msgs::DetectionArrayConstPtr& detections);

protected:
	bool use_image_;
	bool use_point_cloud_;
	bool use_masks_;
	bool use_input_detections_;
	int queue_size_;

	/**
	 * The detector
	 */
    odat::Detector::Ptr detector_;

private:
    std_msgs::Header header_;

	// publishers
	ros::Publisher detections_pub_;
};

}

#endif /* DETECTOR_NODELET_H_ */

