
include_directories(../include/object_detection)

rosbuild_add_gtest(shape_matching_test shape_matching_test.cpp)
rosbuild_add_gtest(glcm_test glcm_test.cpp)

target_link_libraries(shape_matching_test object_detection)
target_link_libraries(glcm_test object_detection)

