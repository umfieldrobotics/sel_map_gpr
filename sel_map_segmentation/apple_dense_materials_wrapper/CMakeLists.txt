cmake_minimum_required(VERSION 3.0.2)
project(apple_dense_materials_wrapper)

## Find catkin macros and libraries
find_package(catkin REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

#############################################
## Clone the segmentation network from git ##
#############################################

include(ExternalProject)
find_package(Git REQUIRED)

## Add the CSAIL Semantic Segmentation Network from Git
set(APPLE_DMS_BUILD_DIR   ${CMAKE_BINARY_DIR}/ext_src/apple_dense_materials)
set(APPLE_DMS_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/apple_dense_materials)
set(APPLE_DMS_INCLUDE_DIR ${APPLE_DMS_INSTALL_DIR})
ExternalProject_Add(
  apple_dense_materials
  GIT_REPOSITORY https://github.com/apple/ml-dms-dataset.git
  PREFIX ${APPLE_DMS_BUILD_DIR}
  GIT_TAG 7e8c1c1652e589ed74370934fdda644c495cf34c
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND 
    ${CMAKE_COMMAND} -E copy_directory ${APPLE_DMS_BUILD_DIR}/src/apple_dense_materials ${APPLE_DMS_INSTALL_DIR}/
  TEST_COMMAND ""
)

###################################
## catkin specific configuration ##
###################################
catkin_package()

