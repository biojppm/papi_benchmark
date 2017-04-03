cmake_minimum_required(VERSION 3.4)
project(papi_example)

set(srcs
    b5_papi.hpp
    b5_papi.cpp
    matrix.hpp
    papi.cpp)

find_library(PAPI_LIB papi)
find_path(PAPI_INC papi.h)

find_library(BM_LIB benchmark)
find_path(BM_INC benchmark/benchmark.h)
find_package(Threads REQUIRED)

add_executable(papi_example ${srcs})
target_include_directories(papi_example PUBLIC ${PAPI_INC} ${BM_INC})
target_link_libraries(papi_example PUBLIC ${PAPI_LIB} ${BM_LIB} ${CMAKE_THREAD_LIBS_INIT})
