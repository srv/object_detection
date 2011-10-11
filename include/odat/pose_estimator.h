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

#ifndef POSE_ESTIMATOR_H_
#define POSE_ESTIMATOR_H_

#include <string>
#include <vector>

#include <boost/make_shared.hpp>
#include <opencv2/core/core.hpp>

namespace rein {

class PoseEstimator
{
public:

	PoseEstimator(){};

	PoseEstimator(ModelStorage::Ptr model_storage) : model_storage_(model_storage) {};

	/**
	 * \brief Set input image for the pose estimator.
	 * @param image The input image
	 */
	inline void setImage(const cv::Mat& image)
	{
		image_ = image;
	}


	/**
	 * Sets the point cloud to be used in the detection.
	 * @param point_cloud
	 */
	void setPointCloud(const sensor_msgs::PointCloud2ConstPtr point_cloud)
	{
		point_cloud_ = point_cloud;
	}


	/**
	 * Informs the pose estimator about detections found in the input
	 * image or point cloud.
	 * @param detections
	 */
	void setDetections(const DetectionArrayConstPtr& detections)
	{
		detections_ = detections;
	}


	/**
	 * \brief Run the object pose estimator.
	 */
	virtual void run() = 0;

	/**
	 * Each pose estimator will have an unique name. This returns the name.
	 * @return name
	 */
	virtual std::string getName() = 0;


	/**
	 * Returns the list of poses of the detected objects
	 * @return
	 */
	inline geometry_msgs::PoseArray getPoses() const
	{
		return poses_;
	}

protected:

	/**
	 * The current image
	 */
	cv::Mat image_;

	/**
	 * Input ROIs
	 */
	DetectionArrayConstPtr detections_;

	/**
	 * PointCloud
	 */
	sensor_msgs::PointCloud2ConstPtr point_cloud_;


	/**
	 * Detection results
	 */
	geometry_msgs::PoseArray poses_;

};
typedef boost::shared_ptr<PoseEstimator> PoseEstimatorPtr;
typedef boost::shared_ptr<PoseEstimator const> PoseEstimatorConstPtr;

}


#endif /* POSE_ESTIMATOR_H_ */
