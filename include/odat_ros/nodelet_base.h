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

#ifndef NODELET_BASE_H_
#define NODELET_BASE_H_

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <list>
#include <boost/make_shared.hpp>


namespace odat_ros {

class NodeletBase : public nodelet::Nodelet
{
public:
	/**
	 * This wraps boost::make_shared and also stores the newly created
	 * shared pointer in a class vector.
	 * @return a shared pointer to the newly created object
	 */
	template <typename T>
	boost::shared_ptr<T> make_shared()
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>();
		shared_ptrs_.push_front(ptr);
		return ptr;
	}

	template <typename T, typename A0>
	boost::shared_ptr<T> make_shared(A0 const & a0)
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>(a0);
		shared_ptrs_.push_front(ptr);
		return ptr;
	}

	template <typename T, typename A0, typename A1>
	boost::shared_ptr<T> make_shared(A0 const & a0, A1 const & a1)
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>(a0, a1);
		shared_ptrs_.push_front(ptr);
		return ptr;
	}

	template <typename T, typename A0, typename A1, typename A2>
	boost::shared_ptr<T> make_shared(A0 const & a0, A1 const & a1, A2 const & a2)
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>(a0, a1, a2);
		shared_ptrs_.push_front(ptr);
		return ptr;
	}

	template <typename T, typename A0, typename A1, typename A2, typename A3>
	boost::shared_ptr<T> make_shared(A0 const & a0, A1 const & a1, A2 const & a2, A3 const & a3)
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>(a0, a1, a2, a3);
		shared_ptrs_.push_front(ptr);
		return ptr;
	}

	template <typename T, typename A0, typename A1, typename A2, typename A3, typename A4>
	boost::shared_ptr<T> make_shared(A0 const & a0, A1 const & a1, A2 const & a2, A3 const & a3, A4 const & a4)
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>(a0, a1, a2, a3, a4);
		shared_ptrs_.push_front(ptr);
		return ptr;
	}

	template <typename T, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5>
	boost::shared_ptr<T> make_shared(A0 const & a0, A1 const & a1, A2 const & a2, A3 const & a3, A4 const & a4, A5 const & a5)
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>(a0, a1, a2, a3, a4, a5);
		shared_ptrs_.push_front(ptr);
		return ptr;
	}

	template <typename T, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
	boost::shared_ptr<T> make_shared(A0 const & a0, A1 const & a1, A2 const & a2, A3 const & a3, A4 const & a4, A5 const & a5, A6 const & a6)
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>(a0, a1, a2, a3, a4, a5, a6);
		shared_ptrs_.push_front(ptr);
		return ptr;
	}

	template <typename T, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
	boost::shared_ptr<T> make_shared(A0 const & a0, A1 const & a1, A2 const & a2, A3 const & a3, A4 const & a4, A5 const & a5, A6 const & a6, A7 const & a7)
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>(a0, a1, a2, a3, a4, a5, a6, a7);
		shared_ptrs_.push_front(ptr);
		return ptr;
	}

	template <typename T, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8>
	boost::shared_ptr<T> make_shared(A0 const & a0, A1 const & a1, A2 const & a2, A3 const & a3, A4 const & a4, A5 const & a5, A6 const & a6, A7 const & a7, A8 const & a8)
	{
		boost::shared_ptr<T> ptr = boost::make_shared<T>(a0, a1, a2, a3, a4, a5, a6, a7, a8);
		shared_ptrs_.push_front(ptr);
		return ptr;
	}



	/**
	 * Initializes the nodelet. Normally you don't need to overwrite this in a subclass,
	 * use childInit(), initReconfigureService() and initReconfigureService() instead.
	 */
	virtual void onInit()
	{
		ros::NodeHandle& private_nh = getMTPrivateNodeHandle();
		childInit(private_nh);
		subscribeTopics(private_nh);
		advertiseTopics(private_nh);
		initConfigureService(private_nh);
	}

	/**
	 * Called on nodelet initialization. Can be used by subclass to perform initialization.
	 * @param nh
	 */
	virtual void childInit(ros::NodeHandle& nh) {};

	/**
	 * Subscribe to the topics of interest. Depending on
	 * some parameters (e.g. use_rois) it can subscribe to different topics.
	 */
	virtual void subscribeTopics(ros::NodeHandle& nh) = 0;

	/**
	 * Advertise the topics this nodelet publishes.
	 */
	virtual void advertiseTopics(ros::NodeHandle& nh) = 0;

	/**
	 * Sets up the dynamic reconfigure service.
	 */
	// not making this pure virtual since not all nodelets must use dynamic-reconfigure
	virtual void initConfigureService(ros::NodeHandle& nh) {}


	/**
	 * Checks to see if anybody is interested in the results.
	 * @return True is somebody subscribed to results
	 */
	virtual bool resultsNeeded() { return true; }

	/**
	 * This does the actual heavy-lifting
	 */
	virtual void runNodelet() = 0;

	/**
	 * Publishes computation results
	 */
	virtual void publishResults() = 0;


protected:
	/**
	 * We use this vector to keep shared pointers to various ros objects (message filters, subscribers),
	 * to prevent polluting the class namespace with objects we don't use directly. Clearing this
	 * vector will delete all those objects
	 */
	std::list< boost::shared_ptr<void> > shared_ptrs_;
};

}


#endif /* NODELET_BASE_H_ */
