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

# object detection and tracking library
rosbuild_add_library(odat
    src/odat/fs_model_storage.cpp
    src/odat/detection.cpp
    src/odat/detector.cpp
    )
rosbuild_link_boost(odat serialization)

# adapter to ros
rosbuild_add_library(odat_ros
    src/odat_ros/conversions.cpp
    )
target_link_libraries(odat_ros odat)

# object detection library
rosbuild_add_library(${PROJECT_NAME}
    src/object_detection/shape_processing.cpp
    src/object_detection/shape_matching.cpp
    src/object_detection/histogram_utilities.cpp 
    src/object_detection/color_detector.cpp
    src/object_detection/shape_detector.cpp
    src/object_detection/clipper.cpp
    )
target_link_libraries(${PROJECT_NAME} odat)

rosbuild_add_executable(detection_display src/nodes/detection_display.cpp)
rosbuild_link_boost(detection_display signals)

rosbuild_add_executable(trainer_node src/nodes/trainer_node.cpp)
target_link_libraries(trainer_node odat_ros ${PROJECT_NAME})

rosbuild_add_executable(color_shape_detector_node src/nodes/color_shape_detector_node.cpp)
rosbuild_add_executable(features3d2d_matching_detector_node src/nodes/features3d2d_matching_detector_node.cpp)
rosbuild_add_executable(features2d2d_matching_detector_node src/nodes/features2d2d_matching_detector_node.cpp)
rosbuild_add_executable(features2d3d_matching_detector_node src/nodes/features2d3d_matching_detector_node.cpp)
rosbuild_link_boost(features2d3d_matching_detector_node signals)
rosbuild_add_executable(detection_tracker src/nodes/detection_tracker.cpp)
rosbuild_link_boost(detection_tracker signals)
target_link_libraries(features2d2d_matching_detector_node odat_ros)
target_link_libraries(color_shape_detector_node ${PROJECT_NAME} odat_ros)
rosbuild_link_boost(features2d3d_matching_detector_node signals)

rosbuild_add_executable(features_to_pcd src/tools/features_to_pcd.cpp)

rosbuild_add_gtest(shape_matching_test test/shape_matching_test.cpp)
rosbuild_add_gtest(shape_processing_test test/shape_processing_test.cpp)
target_link_libraries(shape_matching_test ${PROJECT_NAME})
target_link_libraries(shape_processing_test ${PROJECT_NAME})

rosbuild_add_executable(keypoint_detector_node src/nodes/keypoint_detector_node.cpp)

