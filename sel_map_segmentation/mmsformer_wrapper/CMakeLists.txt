cmake_minimum_required(VERSION 3.0.2)
project(mmsformer_wrapper)

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
set(MMSFORMER_BUILD_DIR   ${CMAKE_BINARY_DIR}/ext_src/mmsformer)
set(MMSFORMER_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/mmsformer)
set(MMSFORMER_INCLUDE_DIR ${MMSFORMER_INSTALL_DIR})
ExternalProject_Add(
  mmsformer
  GIT_REPOSITORY https://github.com/CSIPlab/MMSFormer.git
  PREFIX ${MMSFORMER_BUILD_DIR}
  GIT_TAG b47a5af9a1df53ff68699445a2d0970834acf526
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND 
    ${CMAKE_COMMAND} -E copy_directory ${MMSFORMER_BUILD_DIR}/src/mmsformer ${MMSFORMER_INSTALL_DIR}/
  TEST_COMMAND ""
)

###################################
## catkin specific configuration ##
###################################
catkin_package()

