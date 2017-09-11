cmake_minimum_required(VERSION 2.8)
project(Superresolution)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

find_package(CUDA REQUIRED)
find_package(FreeImage REQUIRED)
find_package(CudaLBFGS REQUIRED)

include(CheckComputeCapability.cmake)

include_directories(${CUDALBFGS_INCLUDE_DIRS})
include_directories(${FreeImage_INCLUDE_DIRS})

# ---

option(SUPERRES_ERROR_CHECKING "Enable CUDA error checking for the superres project.")
if (SUPERRES_ERROR_CHECKING)
	add_definitions(-DSUPERRES_ERROR_CHECKING)
endif()

option(SUPERRES_TIMING "Enable timing for the superres project.")
if (SUPERRES_TIMING)
	add_definitions(-DSUPERRES_TIMING)
endif()

option(SUPERRES_STORE_TRANSPOSE "Store transpose of system matrix for faster A^Tx multiplication." ON)

if (SUPERRES_STORE_TRANSPOSE)
	add_definitions(-DSUPERRES_STORE_TRANSPOSE)
endif()

# ---

file(GLOB CU_FILES *.cu)
file(GLOB H_FILES *.h)
file(GLOB CPP_FILES *.cpp)

source_group("CUDA Sources" FILES ${CU_FILES})
source_group("CPP Sources" FILES ${CPP_FILES})
source_group("Headers" FILES ${H_FILES})

if (NOT DEFINED CUDA_cusparse_LIBRARY OR NOT ${CUDA_cusparse_LIBRARY})
	find_library(CUDA_cusparse_LIBRARY NAMES cusparse HINTS /usr/local/cuda/lib)
endif()

cuda_add_executable(superres ${CU_FILES} ${CPP_FILES} ${H_FILES})
target_link_libraries(superres 
	${CUDALBFGS_LIBRARIES} ${FreeImage_LIBRARY}
	${CUDA_cusparse_LIBRARY})
cuda_add_cublas_to_target(superres)
