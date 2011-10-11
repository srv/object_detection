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

#include "rein/core/types.h"
#include "rein/io/db_model_storage.h"

#include "boost/format.hpp"
#include "boost/make_shared.hpp"

#include "soci.h"
#include "postgresql/soci-postgresql.h"
#include "sqlite3/soci-sqlite3.h"

namespace rein
{

DatabaseModelStorage::DatabaseModelStorage(const std::string& database_type, const std::string& connection_string)
{
	session_ = boost::make_shared<soci::session>();
	if (database_type=="postgresql") {
		session_->open(soci::postgresql,connection_string);
	}
	else if (database_type=="sqlite3") {
		printf("Connection string: %s\n"
				"", connection_string.c_str());
		session_->open(soci::sqlite3,connection_string);
	}
	else {
		throw Exception( (boost::format("Unknown database type: %s") % database_type).str());
	}
}

void DatabaseModelStorage::saveModel(const std::string& name, const std::string& detector, const std::string& model_blob)
{
	// check to see if model is already in database
	(*session_) << "select name from models where name=:name and detector=:detector",
			soci::use(name),soci::use(detector);

	if (session_->got_data()) {
		// already in database, update
		(*session_) << "update models set model_blob=:model_blob where name=:name and detector=:detector",
				soci::use(model_blob),soci::use(name),soci::use(detector);
	}
	else {
		// not in database, insert
		(*session_) << "insert into models(name,detector,model_blob) values(:name,:detector,:model_blob)",
				soci::use(name),soci::use(detector),soci::use(model_blob);
	}
}

bool DatabaseModelStorage::loadModel(const std::string& name, const std::string& detector, std::string& model_blob)
{
	(*session_) << "select model_blob from models where name=:name and detector=:detector",
			soci::use(name),soci::use(detector),soci::into(model_blob);

	return session_->got_data();
}

bool DatabaseModelStorage::getModelList(const std::string& detector, std::vector<std::string>& model_list)
{
	soci::rowset<std::string> rs = (session_->prepare << "select name from models where detector=:detector", soci::use(detector));
	std::copy(rs.begin(), rs.end(), std::back_inserter(model_list));
	return session_->got_data();
}

}


