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

#define BOOST_FILESYSTEM_VERSION 2

#include <boost/format.hpp>
#include <boost/make_shared.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

#include "odat/fs_model_storage.h"
#include "odat/exceptions.h"

using namespace boost::filesystem;

namespace odat
{

FilesystemModelStorage::FilesystemModelStorage(const std::string& directory)
{
	if (!is_directory(directory)) {
		throw Exception( (boost::format("Directory: %s does not exist") % directory).str());
	}
	directory_ = directory;
}

void FilesystemModelStorage::saveModel(const std::string& name, const std::string& detector, const std::string& model_blob)
{
	if (!is_directory(path(directory_)/detector)) {
		create_directory(path(directory_)/detector);
	}
	ofstream ofile(path(directory_)/detector/name);
	std::vector<char> buffer(model_blob.begin(), model_blob.end());
	ofile.write(&buffer[0], buffer.size());
	ofile.close();
}

bool FilesystemModelStorage::loadModel(const std::string& name, const std::string& detector, std::string& model_blob)
{
	ifstream ifile(path(directory_)/detector/name);
	std::vector<char> buffer;
	if (ifile.good()) {
		ifile.seekg(0, std::ios_base::end);
		std::streampos size = ifile.tellg();
		ifile.seekg(0, std::ios_base::beg);
		buffer.resize(size);
		ifile.read(&buffer[0], size);
	}
	else {
		return false;
	}
	ifile.close();

	model_blob.resize(buffer.size());
	std::copy(buffer.begin(), buffer.end(), model_blob.begin());

	return true;
}

bool FilesystemModelStorage::getModelList(const std::string& detector, std::vector<std::string>& model_list)
{
	if ( !exists( path(directory_)/detector ) ) return false;
	directory_iterator end_itr;
	for ( directory_iterator itr( path(directory_)/detector ); itr != end_itr; ++itr ) {
		if ( !is_directory( *itr ) ) {
			model_list.push_back(itr->path().filename());
		}
	}
	return true;
}

}


