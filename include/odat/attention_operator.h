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

#ifndef ATTENTION_OPERATOR_H_
#define ATTENTION_OPERATOR_H_

#include "odat/mask.h"

namespace odat {

class AttentionOperator
{
public:
	/**
	 * \brief Set input image for the detector.
	 * @param image The input image
	 */
	inline void setImage(const cv::Mat& image)
	{
		image_ = image;
	}

	/**
	 * \brief Run the attention operator.
	 */
	virtual void run() = 0;

	/**
	 * Each attention operator will have an unique name.
	 * @return name of the attention operator
	 */
	virtual std::string getName() = 0;


	/**
	 * This returns the list of resulting detection after detect() is called.
	 */
	inline std::vector<Roi> getRois() const
	{
		return rois_;
	}

    typedef boost::shared_ptr<AttentionOperator> Ptr;
    typedef boost::shared_ptr<const AttentionOperator> ConstPtr;

protected:

	/**
	 * The current image
	 */
	cv::Mat image_;

	/**
	 * Detection results
	 */
    std::vector<Roi> rois_;

};

}


#endif /* ATTENTION_OPERATOR_H_ */

