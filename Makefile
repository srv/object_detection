# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/stwirth/robotics/ros/robot_vision/object_detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/stwirth/robotics/ros/robot_vision/object_detection

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/stwirth/robotics/ros/robot_vision/object_detection/CMakeFiles /home/stwirth/robotics/ros/robot_vision/object_detection/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/stwirth/robotics/ros/robot_vision/object_detection/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named ROSBUILD_genmsg_cpp

# Build rule for target.
ROSBUILD_genmsg_cpp: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ROSBUILD_genmsg_cpp
.PHONY : ROSBUILD_genmsg_cpp

# fast build rule for target.
ROSBUILD_genmsg_cpp/fast:
	$(MAKE) -f CMakeFiles/ROSBUILD_genmsg_cpp.dir/build.make CMakeFiles/ROSBUILD_genmsg_cpp.dir/build
.PHONY : ROSBUILD_genmsg_cpp/fast

#=============================================================================
# Target rules for targets named ROSBUILD_genmsg_lisp

# Build rule for target.
ROSBUILD_genmsg_lisp: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ROSBUILD_genmsg_lisp
.PHONY : ROSBUILD_genmsg_lisp

# fast build rule for target.
ROSBUILD_genmsg_lisp/fast:
	$(MAKE) -f CMakeFiles/ROSBUILD_genmsg_lisp.dir/build.make CMakeFiles/ROSBUILD_genmsg_lisp.dir/build
.PHONY : ROSBUILD_genmsg_lisp/fast

#=============================================================================
# Target rules for targets named ROSBUILD_gensrv_cpp

# Build rule for target.
ROSBUILD_gensrv_cpp: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ROSBUILD_gensrv_cpp
.PHONY : ROSBUILD_gensrv_cpp

# fast build rule for target.
ROSBUILD_gensrv_cpp/fast:
	$(MAKE) -f CMakeFiles/ROSBUILD_gensrv_cpp.dir/build.make CMakeFiles/ROSBUILD_gensrv_cpp.dir/build
.PHONY : ROSBUILD_gensrv_cpp/fast

#=============================================================================
# Target rules for targets named ROSBUILD_gensrv_lisp

# Build rule for target.
ROSBUILD_gensrv_lisp: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ROSBUILD_gensrv_lisp
.PHONY : ROSBUILD_gensrv_lisp

# fast build rule for target.
ROSBUILD_gensrv_lisp/fast:
	$(MAKE) -f CMakeFiles/ROSBUILD_gensrv_lisp.dir/build.make CMakeFiles/ROSBUILD_gensrv_lisp.dir/build
.PHONY : ROSBUILD_gensrv_lisp/fast

#=============================================================================
# Target rules for targets named clean-test-results

# Build rule for target.
clean-test-results: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 clean-test-results
.PHONY : clean-test-results

# fast build rule for target.
clean-test-results/fast:
	$(MAKE) -f CMakeFiles/clean-test-results.dir/build.make CMakeFiles/clean-test-results.dir/build
.PHONY : clean-test-results/fast

#=============================================================================
# Target rules for targets named object_detection

# Build rule for target.
object_detection: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 object_detection
.PHONY : object_detection

# fast build rule for target.
object_detection/fast:
	$(MAKE) -f CMakeFiles/object_detection.dir/build.make CMakeFiles/object_detection.dir/build
.PHONY : object_detection/fast

#=============================================================================
# Target rules for targets named object_detection_test

# Build rule for target.
object_detection_test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 object_detection_test
.PHONY : object_detection_test

# fast build rule for target.
object_detection_test/fast:
	$(MAKE) -f CMakeFiles/object_detection_test.dir/build.make CMakeFiles/object_detection_test.dir/build
.PHONY : object_detection_test/fast

#=============================================================================
# Target rules for targets named rosbuild_precompile

# Build rule for target.
rosbuild_precompile: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rosbuild_precompile
.PHONY : rosbuild_precompile

# fast build rule for target.
rosbuild_precompile/fast:
	$(MAKE) -f CMakeFiles/rosbuild_precompile.dir/build.make CMakeFiles/rosbuild_precompile.dir/build
.PHONY : rosbuild_precompile/fast

#=============================================================================
# Target rules for targets named rosbuild_premsgsrvgen

# Build rule for target.
rosbuild_premsgsrvgen: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rosbuild_premsgsrvgen
.PHONY : rosbuild_premsgsrvgen

# fast build rule for target.
rosbuild_premsgsrvgen/fast:
	$(MAKE) -f CMakeFiles/rosbuild_premsgsrvgen.dir/build.make CMakeFiles/rosbuild_premsgsrvgen.dir/build
.PHONY : rosbuild_premsgsrvgen/fast

#=============================================================================
# Target rules for targets named rospack_genmsg

# Build rule for target.
rospack_genmsg: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rospack_genmsg
.PHONY : rospack_genmsg

# fast build rule for target.
rospack_genmsg/fast:
	$(MAKE) -f CMakeFiles/rospack_genmsg.dir/build.make CMakeFiles/rospack_genmsg.dir/build
.PHONY : rospack_genmsg/fast

#=============================================================================
# Target rules for targets named rospack_genmsg_libexe

# Build rule for target.
rospack_genmsg_libexe: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rospack_genmsg_libexe
.PHONY : rospack_genmsg_libexe

# fast build rule for target.
rospack_genmsg_libexe/fast:
	$(MAKE) -f CMakeFiles/rospack_genmsg_libexe.dir/build.make CMakeFiles/rospack_genmsg_libexe.dir/build
.PHONY : rospack_genmsg_libexe/fast

#=============================================================================
# Target rules for targets named rospack_gensrv

# Build rule for target.
rospack_gensrv: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rospack_gensrv
.PHONY : rospack_gensrv

# fast build rule for target.
rospack_gensrv/fast:
	$(MAKE) -f CMakeFiles/rospack_gensrv.dir/build.make CMakeFiles/rospack_gensrv.dir/build
.PHONY : rospack_gensrv/fast

#=============================================================================
# Target rules for targets named test

# Build rule for target.
test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test
.PHONY : test

# fast build rule for target.
test/fast:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/build
.PHONY : test/fast

#=============================================================================
# Target rules for targets named test-future

# Build rule for target.
test-future: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test-future
.PHONY : test-future

# fast build rule for target.
test-future/fast:
	$(MAKE) -f CMakeFiles/test-future.dir/build.make CMakeFiles/test-future.dir/build
.PHONY : test-future/fast

#=============================================================================
# Target rules for targets named test-results

# Build rule for target.
test-results: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test-results
.PHONY : test-results

# fast build rule for target.
test-results/fast:
	$(MAKE) -f CMakeFiles/test-results.dir/build.make CMakeFiles/test-results.dir/build
.PHONY : test-results/fast

#=============================================================================
# Target rules for targets named test-results-run

# Build rule for target.
test-results-run: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test-results-run
.PHONY : test-results-run

# fast build rule for target.
test-results-run/fast:
	$(MAKE) -f CMakeFiles/test-results-run.dir/build.make CMakeFiles/test-results-run.dir/build
.PHONY : test-results-run/fast

#=============================================================================
# Target rules for targets named tests

# Build rule for target.
tests: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 tests
.PHONY : tests

# fast build rule for target.
tests/fast:
	$(MAKE) -f CMakeFiles/tests.dir/build.make CMakeFiles/tests.dir/build
.PHONY : tests/fast

# target to build an object file
src/object_detection/histogram_detector.o:
	$(MAKE) -f CMakeFiles/object_detection.dir/build.make CMakeFiles/object_detection.dir/src/object_detection/histogram_detector.o
.PHONY : src/object_detection/histogram_detector.o

# target to preprocess a source file
src/object_detection/histogram_detector.i:
	$(MAKE) -f CMakeFiles/object_detection.dir/build.make CMakeFiles/object_detection.dir/src/object_detection/histogram_detector.i
.PHONY : src/object_detection/histogram_detector.i

# target to generate assembly for a file
src/object_detection/histogram_detector.s:
	$(MAKE) -f CMakeFiles/object_detection.dir/build.make CMakeFiles/object_detection.dir/src/object_detection/histogram_detector.s
.PHONY : src/object_detection/histogram_detector.s

# target to build an object file
src/object_detection_test/main.o:
	$(MAKE) -f CMakeFiles/object_detection_test.dir/build.make CMakeFiles/object_detection_test.dir/src/object_detection_test/main.o
.PHONY : src/object_detection_test/main.o

# target to preprocess a source file
src/object_detection_test/main.i:
	$(MAKE) -f CMakeFiles/object_detection_test.dir/build.make CMakeFiles/object_detection_test.dir/src/object_detection_test/main.i
.PHONY : src/object_detection_test/main.i

# target to generate assembly for a file
src/object_detection_test/main.s:
	$(MAKE) -f CMakeFiles/object_detection_test.dir/build.make CMakeFiles/object_detection_test.dir/src/object_detection_test/main.s
.PHONY : src/object_detection_test/main.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... ROSBUILD_genmsg_cpp"
	@echo "... ROSBUILD_genmsg_lisp"
	@echo "... ROSBUILD_gensrv_cpp"
	@echo "... ROSBUILD_gensrv_lisp"
	@echo "... clean-test-results"
	@echo "... edit_cache"
	@echo "... object_detection"
	@echo "... object_detection_test"
	@echo "... rebuild_cache"
	@echo "... rosbuild_precompile"
	@echo "... rosbuild_premsgsrvgen"
	@echo "... rospack_genmsg"
	@echo "... rospack_genmsg_libexe"
	@echo "... rospack_gensrv"
	@echo "... test"
	@echo "... test-future"
	@echo "... test-results"
	@echo "... test-results-run"
	@echo "... tests"
	@echo "... src/object_detection/histogram_detector.o"
	@echo "... src/object_detection/histogram_detector.i"
	@echo "... src/object_detection/histogram_detector.s"
	@echo "... src/object_detection_test/main.o"
	@echo "... src/object_detection_test/main.i"
	@echo "... src/object_detection_test/main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

