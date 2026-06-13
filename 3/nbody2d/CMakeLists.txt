cmake_minimum_required(VERSION 3.18)
project(nbody2d)

# Some IDEs need the compile commands to handle CMake projects properly.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI)

set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -Wall -std=c99 -fopenmp -D_POSIX_C_SOURCE=200809L")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Adding utility options
option(PPP_DEBUG "Enable debug mode (no optimizations)." off)
if (PPP_DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -fno-omit-frame-pointer")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fomit-frame-pointer -DNDEBUG")
endif (PPP_DEBUG)

include_directories(
  include
  ${MPI_C_INCLUDE_PATH}
  )

add_subdirectory(src)
