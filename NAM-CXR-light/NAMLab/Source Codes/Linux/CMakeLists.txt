cmake_minimum_required(VERSION 3.15)
project(NAMLab)

find_package(OpenCV REQUIRED)
find_package(Matlab REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Matlab_INCLUDE_DIRS})
include_directories(Sources)

set(CMAKE_CXX_STANDARD 14)

file(GLOB SOURCES
        Sources/*.cpp
        Sources/*.h
        )

message(${SOURCES})

add_executable(run_demo run_demo.cpp ${SOURCES})
add_executable(run_dataset run_dataset.cpp ${SOURCES})

target_link_libraries(run_demo ${OpenCV_LIBS})
target_link_libraries(run_demo ${Matlab_LIBRARIES} ${Matlab_ROOT_DIR}/bin/glnxa64/libmat.so)
target_link_libraries(run_dataset ${OpenCV_LIBS})
target_link_libraries(run_dataset ${Matlab_LIBRARIES} ${Matlab_ROOT_DIR}/bin/glnxa64/libmat.so)
