cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

set(CMAKE_CXX_STANDARD 14)
# set(Torch_DIR "/home/cuimingdi/Project/ORB_SLAM3/libtorch/share/cmake/Torch") 

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch/")
find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})


# test-libtorch/SPDetector.cpp test-libtorch/SPDetector.hpp test-libtorch/SuperPoint.cpp test-libtorch/SuperPoint.hpp
add_library(sp_lib SHARED ${PROJECT_SOURCE_DIR}/SPDetector.cpp ${PROJECT_SOURCE_DIR}/SPDetector.hpp ${PROJECT_SOURCE_DIR}/SuperPoint.cpp ${PROJECT_SOURCE_DIR}/SuperPoint.hpp)
target_link_libraries(sp_lib ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(example-app example-app.cpp)
target_link_libraries(example-app ${TORCH_LIBRARIES} ${OpenCV_LIBS} sp_lib)