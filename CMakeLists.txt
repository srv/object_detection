cmake_minimum_required(VERSION 2.8.3)
project(object_detection)

find_package(catkin REQUIRED COMPONENTS 
    std_msgs 
    sensor_msgs
    nav_msgs 
    cv_bridge 
    roscpp
    image_transport
    image_geometry
    camera_info_manager
    pcl_ros 
    tf 
    nodelet
    vision_msgs
    feature_extraction)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options system filesystem serialization signals system)

catkin_package(CATKIN_DEPENDS feature_extraction feature_matching)

include_directories(${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} include)

# LIBRARIES
add_library(odat
    src/odat/fs_model_storage.cpp
    src/odat/detection.cpp
    src/odat/detector.cpp
    )
target_link_libraries(odat ${Boost_LIBRARIES})

add_library(odat_ros src/odat_ros/conversions.cpp)
add_dependencies(odat_ros vision_msgs_gencpp)
target_link_libraries(odat_ros odat)

add_library(${PROJECT_NAME}
    src/object_detection/shape_processing.cpp
    src/object_detection/shape_matching.cpp
    src/object_detection/histogram_utilities.cpp 
    src/object_detection/color_detector.cpp
    src/object_detection/shape_detector.cpp
    src/object_detection/clipper.cpp
    )
target_link_libraries(${PROJECT_NAME} odat)

add_library(${PROJECT_NAME}_ros src/object_detection_ros/detector_nodelet.cpp)
add_dependencies(${PROJECT_NAME} vision_msgs_gencpp)
target_link_libraries(${PROJECT_NAME}_ros ${PROJECT_NAME} odat_ros)

# EXECUTABLES
add_executable(detection_display src/nodes/detection_display.cpp)
add_dependencies(detection_display vision_msgs_gencpp)
target_link_libraries(detection_display ${Boost_LIBRARIES} ${catkin_LIBRARIES})

add_executable(trainer_node src/nodes/trainer_node.cpp)
add_dependencies(trainer_node vision_msgs_gencpp)
target_link_libraries(trainer_node odat_ros ${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(feature_matching_detector_node src/nodes/feature_matching_detector_node.cpp)
add_dependencies(feature_matching_detector_node vision_msgs_gencpp)
target_link_libraries(feature_matching_detector_node ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(features2d_matching_detector_node src/nodes/features2d_matching_detector_node.cpp)
add_dependencies(features2d_matching_detector_node vision_msgs_gencpp)
target_link_libraries(features2d_matching_detector_node odat_ros ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

add_executable(features2d3d_matching_detector_node src/nodes/features2d3d_matching_detector_node.cpp)
add_dependencies(features2d3d_matching_detector_node vision_msgs_gencpp)
target_link_libraries(features2d3d_matching_detector_node ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(detection_tracker src/nodes/detection_tracker.cpp)
add_dependencies(detection_tracker vision_msgs_gencpp)
target_link_libraries(detection_tracker ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(train_from_disk src/nodes/train_from_disk.cpp)
target_link_libraries(train_from_disk odat_ros ${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(features_to_pcd src/tools/features_to_pcd.cpp)
target_link_libraries(features_to_pcd ${OpenCV_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES})

catkin_add_gtest(shape_matching_test test/shape_matching_test.cpp)
target_link_libraries(shape_matching_test ${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

catkin_add_gtest(shape_processing_test test/shape_processing_test.cpp)
target_link_libraries(shape_processing_test ${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(keypoint_detector_node src/nodes/keypoint_detector_node.cpp)
add_dependencies(keypoint_detector_node vision_msgs_gencpp)
target_link_libraries(keypoint_detector_node ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

