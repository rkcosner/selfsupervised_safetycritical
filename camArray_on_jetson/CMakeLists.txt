cmake_minimum_required(VERSION 2.8.11)
project(camArray)


find_package(OpenCV  4 REQUIRED PATHS /usr/lib/aarch64-linux-gnu/cmake/opencv4 NO_DEFAULT_PATHS)


find_package(catkin REQUIRED COMPONENTS 
                roscpp 
                image_transport 
                rospy 
                cv_bridge 
                message_generation
                uncertainty_estimator
)


catkin_package( CATKIN_DEPENDS message_runtime)
set (CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} -march=armv8-a+fp+simd+crypto+crc")


set(VISION_SRC_DIR include)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${VISION_SRC_DIR})

add_executable(videoReader 
                  src/videoReader.cpp
)

add_dependencies(videoReader ${catkin_EXPORTED_TARGETS})

target_link_libraries(videoReader 
                        ${catkin_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT}
                        ${OpenCV_LIBS}
)

catkin_install_python(PROGRAMS
  src/commander.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
