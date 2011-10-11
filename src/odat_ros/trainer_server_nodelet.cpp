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

#include "rein/nodelets/trainer_server.h"
#include "rein/nodelets/type_conversions.h"


REGISTER_TYPE_CONVERSION(rein::TrainInstance::Request, rein::TrainingData,
		(image, image)
		(mask.roi, roi)
		(mask.mask, mask)
		(background, background)
)


namespace rein
{


/**
* Instantiates a TrainerServer with a Trainable object.
* @param nh
* @param trainee
* @return
*/
TrainerServer::TrainerServer(ros::NodeHandle& nh, TrainablePtr trainee) : trainee_(trainee)
{
	start_training_service_ = nh.advertiseService("start_training",&TrainerServer::startTraining, this);
	train_instance_service_ = nh.advertiseService("train_instance",&TrainerServer::trainInstance, this);
	save_model_service_ = nh.advertiseService("end_training",&TrainerServer::endTraining, this);
}

/**
 * Start Training service call. Must be called before training a new
 * model is started.
 * @param req
 * @param resp
 * @return
 */
bool TrainerServer::startTraining(StartTraining::Request& req, StartTraining::Response& resp)
{
	try {
		ROS_INFO("TrainerServer: starting training for class: %s", req.name.c_str());
		trainee_->startTraining(req.name);
	}
	catch (const Exception& e) {
		ROS_ERROR("TrainerServer: %s",e.what());
		return false;
	}

	return true;
}

/**
 * TrainInstance service call. Is called for each instance that the
 * model is trained with.
 * @param req
 * @param resp
 * @return
 */
bool TrainerServer::trainInstance(TrainInstance::Request& req, TrainInstance::Response& resp)
{
	try {
		TrainingData training_data;
		convert(req, training_data);
		trainee_->trainInstance(req.name, training_data);
	}
	catch (const Exception& e) {
		ROS_ERROR("TrainerServer: %s",e.what());
		return false;
	}

	return true;
}

/**
 * SaveModel service call. Is called for saving the model at the end
 * of the training.
 * @param req
 * @param resp
 * @return
 */
bool TrainerServer::endTraining(EndTraining::Request& req, EndTraining::Response& resp)
{
	try {
		trainee_->endTraining(req.name);
		ROS_INFO("TrainerServer: finished training model: %s", req.name.c_str());
	}
	catch (const Exception& e) {
		ROS_ERROR("TrainerServer: %s",e.what());
		return false;
	}

	return true;
}

}

