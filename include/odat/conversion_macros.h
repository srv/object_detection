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

#ifndef CONVERSIONS_H_
#define CONVERSIONS_H_

#include <vector>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/cat.hpp>

template <typename From, typename To>
inline void convert(const From& a, To& b) { b = a; }

template <typename From, typename To>
inline void convert(const std::vector<From>& a, std::vector<To>& b)
{
	b.resize(a.size());
	for (size_t i=0;i<a.size();++i) {
		convert(a[i],b[i]);
	}
}

#define REGISTER_TYPE_CONVERSION_H(type_a,type_b)						\
		void convert(const type_a& a, type_b& b);                     \
	    void convert(const type_b& a, type_a& b);

#define REGISTER_TYPE_CONVERSION(type_a,type_b,seq)                     \
    REGISTER_TYPE_CONVERSION_I(type_a,type_b,                           \
            BOOST_PP_CAT(REGISTER_TYPE_CONVERSION_X seq,0))

#define REGISTER_TYPE_CONVERSION_X(name_a,name_b)                       \
    ((name_a,name_b)) REGISTER_TYPE_CONVERSION_Y
#define REGISTER_TYPE_CONVERSION_Y(name_a,name_b)                       \
    ((name_a,name_b)) REGISTER_TYPE_CONVERSION_X
#define REGISTER_TYPE_CONVERSION_X0
#define REGISTER_TYPE_CONVERSION_Y0


#define REGISTER_TYPE_CONVERSION_I(type_a,type_b,seq)                   \
    void convert(const type_a& a, type_b& b)                     \
    {                                                                   \
        BOOST_PP_SEQ_FOR_EACH(REGISTER_FIELD_CONVERSION_F,_,seq)        \
    }                                                                   \
    void convert(const type_b& a, type_a& b)                     \
    {                                                                   \
        BOOST_PP_SEQ_FOR_EACH(REGISTER_FIELD_CONVERSION_B,_,seq)        \
    }                                                                   \

#define REGISTER_FIELD_CONVERSION_F(r, data, elem)\
        convert(a.BOOST_PP_TUPLE_ELEM(2, 0, elem),b.BOOST_PP_TUPLE_ELEM(2, 1, elem));

#define REGISTER_FIELD_CONVERSION_B(r, data, elem)\
        convert(a.BOOST_PP_TUPLE_ELEM(2, 1, elem),b.BOOST_PP_TUPLE_ELEM(2, 0, elem));



#endif /* CONVERSIONS_H_ */
