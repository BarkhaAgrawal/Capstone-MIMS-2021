# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.19.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.19.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp"

# Include any dependencies generated for this target.
include CMakeFiles/sptest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sptest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sptest.dir/flags.make

CMakeFiles/sptest.dir/src/graph.cc.o: CMakeFiles/sptest.dir/flags.make
CMakeFiles/sptest.dir/src/graph.cc.o: src/graph.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sptest.dir/src/graph.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sptest.dir/src/graph.cc.o -c "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/src/graph.cc"

CMakeFiles/sptest.dir/src/graph.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sptest.dir/src/graph.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/src/graph.cc" > CMakeFiles/sptest.dir/src/graph.cc.i

CMakeFiles/sptest.dir/src/graph.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sptest.dir/src/graph.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/src/graph.cc" -o CMakeFiles/sptest.dir/src/graph.cc.s

CMakeFiles/sptest.dir/tests/test_main.cc.o: CMakeFiles/sptest.dir/flags.make
CMakeFiles/sptest.dir/tests/test_main.cc.o: tests/test_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sptest.dir/tests/test_main.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sptest.dir/tests/test_main.cc.o -c "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/tests/test_main.cc"

CMakeFiles/sptest.dir/tests/test_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sptest.dir/tests/test_main.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/tests/test_main.cc" > CMakeFiles/sptest.dir/tests/test_main.cc.i

CMakeFiles/sptest.dir/tests/test_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sptest.dir/tests/test_main.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/tests/test_main.cc" -o CMakeFiles/sptest.dir/tests/test_main.cc.s

CMakeFiles/sptest.dir/tests/graph_test.cc.o: CMakeFiles/sptest.dir/flags.make
CMakeFiles/sptest.dir/tests/graph_test.cc.o: tests/graph_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sptest.dir/tests/graph_test.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sptest.dir/tests/graph_test.cc.o -c "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/tests/graph_test.cc"

CMakeFiles/sptest.dir/tests/graph_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sptest.dir/tests/graph_test.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/tests/graph_test.cc" > CMakeFiles/sptest.dir/tests/graph_test.cc.i

CMakeFiles/sptest.dir/tests/graph_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sptest.dir/tests/graph_test.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/tests/graph_test.cc" -o CMakeFiles/sptest.dir/tests/graph_test.cc.s

# Object files for target sptest
sptest_OBJECTS = \
"CMakeFiles/sptest.dir/src/graph.cc.o" \
"CMakeFiles/sptest.dir/tests/test_main.cc.o" \
"CMakeFiles/sptest.dir/tests/graph_test.cc.o"

# External object files for target sptest
sptest_EXTERNAL_OBJECTS =

sptest: CMakeFiles/sptest.dir/src/graph.cc.o
sptest: CMakeFiles/sptest.dir/tests/test_main.cc.o
sptest: CMakeFiles/sptest.dir/tests/graph_test.cc.o
sptest: CMakeFiles/sptest.dir/build.make
sptest: CMakeFiles/sptest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable sptest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sptest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sptest.dir/build: sptest

.PHONY : CMakeFiles/sptest.dir/build

CMakeFiles/sptest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sptest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sptest.dir/clean

CMakeFiles/sptest.dir/depend:
	cd "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp" "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp" "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp" "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp" "/Users/barkha/Documents/UCB/Spring 2021/Capstone/spatial_queue-master/projects/bolinas_civic/sp/CMakeFiles/sptest.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/sptest.dir/depend
