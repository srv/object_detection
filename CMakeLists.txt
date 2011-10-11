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
    src/object_detection/textured_parts_classifier.cpp
    src/object_detection/histogram_utilities.cpp 
    src/object_detection/utilities.cpp
    src/object_detection/statistics.cpp
    src/object_detection/shape_processing.cpp
    src/object_detection/shape_matching.cpp
    src/object_detection/clipper.cpp
    src/object_detection/glcm.cpp
    src/object_detection/model.cpp
    src/object_detection/transformation_calculator.cpp
   )

set(NEW_LIB_SOURCES
    src/rein_object_detection/color_detector.cpp
    src/nodelets/color_detector_nodelet.cpp
   )

# object detection and tracking library
rosbuild_add_library(odat
    src/odat/fs_model_storage.cpp
    src/odat/detection.cpp
    src/odat/detector.cpp
    )
rosbuild_link_boost(odat serialization)

# adapter to ros
rosbuild_add_library(odat_ros
    src/odat_ros/detector_nodelet.cpp
    src/odat_ros/conversions.cpp
    )
target_link_libraries(odat_ros odat)

# object detection library
rosbuild_add_library(${PROJECT_NAME}
    src/object_detection/shape_processing.cpp
    src/object_detection/shape_matching.cpp
    src/object_detection/histogram_utilities.cpp 
    src/object_detection/color_detector.cpp
    src/object_detection/feature_matching_detector.cpp
    src/object_detection/shape_detector.cpp
    src/object_detection/clipper.cpp
    )
target_link_libraries(${PROJECT_NAME} odat)

# ros adapter to object detection library
rosbuild_add_library(${PROJECT_NAME}_ros
    src/object_detection_ros/color_detector_nodelet.cpp
    src/object_detection_ros/shape_detector_nodelet.cpp
    src/object_detection_ros/feature_matching_detector_nodelet.cpp
    )
target_link_libraries(${PROJECT_NAME}_ros ${PROJECT_NAME} odat_ros)

rosbuild_add_executable(trainer src/object_detection_tools/trainer.cpp)
target_link_libraries(trainer ${PROJECT_NAME})

rosbuild_add_executable(detector src/object_detection_tools/detector.cpp)
target_link_libraries(detector ${PROJECT_NAME})

rosbuild_add_executable(detection_display src/odat_ros/tools/detection_display.cpp)
target_link_libraries(detection_display odat_ros)

rosbuild_add_executable(trainer_node src/nodes/trainer_node.cpp)
target_link_libraries(trainer_node odat_ros ${PROJECT_NAME})


rosbuild_add_gtest(shape_matching_test test/shape_matching_test.cpp)
target_link_libraries(shape_matching_test ${PROJECT_NAME})

#rosbuild_add_library(object_detection ${LIB_SOURCES} ${NEW_LIB_SOURCES})
#rosbuild_add_executable(object_detection_test src/object_detection_test/main.cpp)
#rosbuild_add_executable(pixel_feature_extraction src/pixel_feature_extraction/main.cpp)
#rosbuild_add_executable(glcm_extraction src/glcm_extraction/main.cpp)
#target_link_libraries(object_detection boost_program_options sqlite3 pq)

#target_link_libraries(object_detection_test object_detection)
#target_link_libraries(pixel_feature_extraction object_detection)
#target_link_libraries(glcm_extraction object_detection)

#include_directories(include/object_detection)
#include_directories(include/rein_object_detection)

#rosbuild_add_executable(object_detector_node src/nodes/object_detector_node.cpp)
#rosbuild_add_executable(trainer_node src/nodes/trainer_node.cpp)
#rosbuild_add_executable(stereo_trainer_node src/nodes/stereo_trainer_node.cpp)
#rosbuild_add_executable(keypoint_detector_node src/nodes/keypoint_detector_node.cpp)
#rosbuild_add_executable(detection_display_node src/nodes/detection_display_node.cpp)
#rosbuild_add_executable(movement_calculator_node src/nodes/movement_calculator_node.cpp)
#rosbuild_add_executable(model_fitting_node src/nodes/model_fitting_node.cpp)
#rosbuild_add_executable(joiner src/tools/joiner.cpp)
#rosbuild_add_executable(transformation_estimator src/tools/transformation_estimator.cpp)
#rosbuild_add_executable(transformation_error_calculator src/tools/transformation_error_calculator.cpp)
#rosbuild_add_executable(feature_quality src/tools/feature_quality.cpp)
#rosbuild_add_executable(sample_distance_threshold_computation src/tools/sample_distance_threshold_computation.cpp)

#target_link_libraries(color_detector_trainer object_detection)
#target_link_libraries(object_detector_node object_detection)
#target_link_libraries(trainer_node object_detection)
#target_link_libraries(stereo_trainer_node object_detection)
#target_link_libraries(keypoint_detector_node object_detection)
#target_link_libraries(detection_display_node object_detection)

