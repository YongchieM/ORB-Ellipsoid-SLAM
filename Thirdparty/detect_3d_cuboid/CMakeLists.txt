cmake_minimum_required(VERSION 2.8)
project(detect_3d_cuboid)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( CMAKE_BUILD_TYPE Release)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
## System dependencies are found with CMake's conventions

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}  
  ${PROJECT_SOURCE_DIR}/Thirdparty/tictoc_profiler/include
  ${PROJECT_SOURCE_DIR}/Thirdparty/line_lbd/include
)

## Declare a C++ library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME}  SHARED
src/box_proposal_detail.cpp src/object_3d_util.cpp src/matrix_utils.cpp  
)

target_link_libraries(${PROJECT_NAME}   ${OpenCV_LIBRARIES} ${EIGEN3_LIBRARIES} ${Boost_SYSTEM_LIBRARY}
${PROJECT_SOURCE_DIR}/Thirdparty/line_lbd/lib/libline_lbd_lib.so
${PROJECT_SOURCE_DIR}/Thirdparty/tictoc_profiler/lib/libtictoc_profiler.so
)

## Declare a C++ executable
add_executable(detect_3d_cuboid_node src/main.cpp)

target_link_libraries(detect_3d_cuboid_node  
   ${PROJECT_NAME}   ${OpenCV_LIBRARIES} ${EIGEN3_LIBRARIES} 
)

# add_executable(detect_3d_cuboid_chair src/main_chair.cpp)
# target_link_libraries(detect_3d_cuboid_chair  ${PROJECT_NAME}
#   ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES}
# )

