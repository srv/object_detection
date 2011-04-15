cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} example/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

set(LIB_SOURCES 
    src/object_detection/detector.cpp
    src/object_detection/detection.cpp
    src/object_detection/object_parts_detector.cpp
    src/object_detection/histogram_based_parts_classifier.cpp
    src/object_detection/colored_parts_classifier.cpp
    src/object_detection/histogram_utilities.cpp 
    src/object_detection/utilities.cpp
    src/object_detection/statistics.cpp
   )


rosbuild_add_library(object_detection ${LIB_SOURCES})
rosbuild_add_executable(object_detection_test src/object_detection_test/main.cpp)
rosbuild_add_executable(histograms src/histograms/main.cpp)

target_link_libraries(object_detection_test object_detection)
target_link_libraries(histograms object_detection)

include_directories(include/object_detection)

