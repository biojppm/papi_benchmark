cmake_minimum_required(VERSION 3.4)

project(papi_extern)

# we'll be using ExternalProject
include(ExternalProject)
# ... and an utility file on top of ExternalProject:
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ExternalProjectUtils.cmake)
    file(DOWNLOAD
        https://raw.githubusercontent.com/biojppm/cmake/master/ExternalProjectUtils.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/ExternalProjectUtils.cmake
        )
endif()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(ExternalProjectUtils)

# with this we can forward the current compile environment
# to each external project
ExternalProject_GetFwdArgs(FWD_ARGS)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

set(BDIR ${CMAKE_BINARY_DIR})
set(IDIR ${CMAKE_INSTALL_PREFIX})
# this is needed for multi-configuration generators.
# see http://stackoverflow.com/questions/24460486/cmake-build-type-not-being-used-in-cmakelists-txt
set(BCMD ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --)

ExternalProject_Add(googlebenchmark
    PREFIX          googlebenchmark
    GIT_REPOSITORY  https://github.com/google/benchmark
    BINARY_DIR      ${BDIR}/googlebenchmark
    INSTALL_DIR     ${IDIR}
    CMAKE_ARGS      ${FWD_ARGS} -DBENCHMARK_ENABLE_TESTING=0
    BUILD_COMMAND   ${BCMD})
