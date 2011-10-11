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

#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <string>
#include <vector>

#include <boost/make_shared.hpp>

#include "odat/detection.h"
#include "odat/model_storage.h"
#include "odat/mask.h"
#include "odat/feature_set.h"

namespace odat {

class Detector
{
public:

	Detector() {};

	Detector(ModelStorage::Ptr model_storage) : model_storage_(model_storage) {};

	/**
	 * \brief Set input image for the detector.
	 * @param image The input image
	 */
	inline void setImage(const cv::Mat& image)
	{
		image_ = image;
	}

	/**
	 * \brief Set camera matrix
	 * @param k camera matrix
	 */
	inline void setCameraMatrix(const cv::Mat& k)
    {
        camera_matrix_k_ = k;
    }

	/**
	 * Sets the regions-of-interest (ROI/masks) where the detector should look.
	 * @param ros_list Array of rectangles representing the regions of interest. If empty
	 * the entire image is used.
	 */
	inline void setMasks(const std::vector<Mask>& masks)
	{
		masks_ = masks;
	}

	/**
	 * Sets the input features
	 * @param feature_set the set of input features
	 */
	inline void setFeatures(const FeatureSet& features)
    {
        features_ = features;
    }

	/**
	 * Set list of detections. This is useful when the detector works like a filter, for chaining detectors
	 * @param ros_list Array of detections.
	 */
	inline void setInputdetections(const std::vector<Detection>& input_detections)
	{
		input_detections_ = input_detections;
	}

	/**
	 * \brief Run the object detector. The detection results are stored in
	 * class member detections_.
	 */
	virtual void detect() = 0;

	/**
	 * Each detector will have an unique name. This returns the detector name.
	 * @return name of the detector
	 */
	virtual std::string getName() = 0;

	/**
	 * Loads pre-trained models for a list of objects.
	 * @param models list of objects to load
	 */
	virtual void loadModels(const std::vector<std::string>& models) = 0;

	/**
	 * Loads pre-trained models for a list of objects.
	 * @param models comma-separated list of objects to load
	 */
	void loadModelList(const std::string& models);

	/**
	 * Loads all available pre-trained models
	 */
	void loadAllModels();

	/**
	* Returns a list containing the names of all loaded models
	*/
    virtual std::vector<std::string> getLoadedModels() const = 0;

	/**
	 * This returns the list of resulting detection after detect() is called.
	 * @return detections array
	 */
	inline std::vector<Detection> getDetections() const
	{
		return detections_;
	}

    typedef boost::shared_ptr<Detector> Ptr;
    typedef boost::shared_ptr<const Detector> ConstPtr;

protected:

	/**
	 * The current image
	 */
	cv::Mat image_;

	/**
	 * The camera matrix
	 */
    cv::Mat camera_matrix_k_;

	/**
	 * Input Masks
	 */
    std::vector<Mask> masks_;

    /**
     * Features
     */
    FeatureSet features_;

	/**
	 * Input ROIs
	 */
    std::vector<Detection> input_detections_;

	/**
	 * Detection results
	 */
    std::vector<Detection> detections_;

	/**
	 * Model storage object
	 */
    ModelStorage::Ptr model_storage_;

};
}


#endif /* DETECTOR_H_ */

