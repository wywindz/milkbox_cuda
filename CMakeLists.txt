cmake_minimum_required(VERSION 2.8)

project(milkbox_cuda)

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /opt/freenect2/)

INCLUDE(/usr/share/cmake-3.5/Modules/FindCUDA.cmake)

find_package(PCL 1.8 REQUIRED PATHS /usr NO_DEFAULT_PATH)
find_package(VTK REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(freenect2 REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
        list(APPEND vtk_flags ${it})
    endif()
endforeach()
foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()


include(${VTK_USE_FILE})
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${freenect2_INCLUDE_DIR}
    "/usr/local/include"
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

link_directories(
    ${PCL_LIBRARY_DIRS}
    "/opt/freenect2/lib/")

add_definitions(${PCL_DEFINITIONS})


CUDA_ADD_EXECUTABLE(${PROJECT_NAME}
#    app/main.cpp
    test/test.cpp
    grabber/kinect2grabber.cpp
    mb_cuda/common/point_types.h
    mb_cuda/io/pcl_thrust.cpp
    mb_cuda/io/pcl_thrust.h
    mb_cuda/io/host_device.h
    mb_cuda/io/host_device.cu
    mb_cuda/filters/pass_through.h
    mb_cuda/filters/pass_through.cu
    mb_cuda/filters/voxel_grid.h
    mb_cuda/filters/voxel_grid.cu
#    mb_cuda/filters/filter.h
#    mb_cuda/common/point_cloud.h
#    mb_cuda/io/host_device.h
#    mb_cuda/io/host_device.cu
    )

#target_link_libraries(${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME}
    freenect2
    ${OpenCV_LIBRARIES}
    ${VTK_LIBRARIES}
    ${PCL_LIBRARIES}
)
