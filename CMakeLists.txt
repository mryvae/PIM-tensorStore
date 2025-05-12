cmake_minimum_required(VERSION 3.14)
project(PIM-tensorStore)

set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT NR_TASKLETS)
    set(NR_TASKLETS 16)
    message(WARNING "NR_TASKLETS is " ${NR_TASKLETS})
endif()

set(UPMEM_SDK_DIR "$ENV{UPMEM_SDK_DIR}")
if(NOT UPMEM_SDK_DIR)
    message(FATAL_ERROR "environment variable UPMEM_SDK_DIR is not set")
    exit()
endif()

message("UPMEM_SDK_DIR: "${UPMEM_SDK_DIR})

set(UPMEM_C_COMPILER ${UPMEM_SDK_DIR}/bin/dpu-upmem-dpurte-clang)

if(NOT EXECUTABLE_OUTPUT_PATH)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
endif()

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY )
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_SOURCE_DIR}/bin)
endif()

file(GLOB_RECURSE DPU_SOURCES "dpu/*.c")
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/host/util UTIL_SOURCES)

## build dpu binary
file(GLOB_RECURSE DPU_HEADERS "dpu/*.h")
file(GLOB UTIL_HEADERS "host/util/*.h")

set(DPU_TASK_NAME dpu_task)
add_custom_target(${DPU_TASK_NAME} ALL
    COMMAND ${UPMEM_C_COMPILER} -O3 -fgnu89-inline
            -DNR_TASKLETS=${NR_TASKLETS}
            -DSTACK_SIZE_DEFAULT=1024
            -I${CMAKE_CURRENT_SOURCE_DIR}/dpu/ops
            -I${CMAKE_CURRENT_SOURCE_DIR}/dpu/sto
            -I${CMAKE_CURRENT_SOURCE_DIR}/dpu/util
            -I${CMAKE_CURRENT_SOURCE_DIR}/host/mm
            -I${CMAKE_CURRENT_SOURCE_DIR}/host/msg
            -I${CMAKE_CURRENT_SOURCE_DIR}/host/util
            ${DPU_SOURCES}
            ${UTIL_SOURCES}
            -o ${EXECUTABLE_OUTPUT_PATH}/${DPU_TASK_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${DPU_SOURCES} ${UTIL_SOURCES} 
            ${DPU_HEADERS} ${UTIL_HEADERS}
    VERBATIM)

install(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${DPU_TASK_NAME} DESTINATION bin)

# build host
set(UPMEM_HOST_NAME host_app)
set(C_OPTIONS -O3 -std=gnu99 -fgnu89-inline -DNR_TASKLETS=${NR_TASKLETS} -pthread)

file(GLOB_RECURSE HOST_SOURCES "host/*.c")
add_executable(${UPMEM_HOST_NAME} ${HOST_SOURCES})
set_target_properties(${UPMEM_HOST_NAME} PROPERTIES COMPILE_OPTIONS "${C_OPTIONS}")
target_include_directories(${UPMEM_HOST_NAME} 
                        PRIVATE ${UPMEM_SDK_DIR}/include/dpu
                        ${CMAKE_CURRENT_SOURCE_DIR}/host/mm
                        ${CMAKE_CURRENT_SOURCE_DIR}/host/msg
                        ${CMAKE_CURRENT_SOURCE_DIR}/host/util)
target_link_directories(${UPMEM_HOST_NAME} PRIVATE ${UPMEM_SDK_DIR}/lib)
target_link_libraries(${UPMEM_HOST_NAME} dpu dpuverbose)

install(TARGETS ${UPMEM_HOST_NAME} RUNTIME DESTINATION bin)

# build host as lib
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/host HOST_LIB_SOURCES)
list(REMOVE_ITEM HOST_LIB_SOURCES main.c)

set(PIM_TENSOR_STORE_LIB_NAME PimTensorStore)

add_library(${PIM_TENSOR_STORE_LIB_NAME} SHARED ${HOST_LIB_SOURCES})
set_target_properties(${PIM_TENSOR_STORE_LIB_NAME} PROPERTIES COMPILE_OPTIONS "${C_OPTIONS}")
target_include_directories(${PIM_TENSOR_STORE_LIB_NAME}
                            PUBLIC ${UPMEM_SDK_DIR}/include/dpu
                            ${CMAKE_CURRENT_SOURCE_DIR}/host/mm
                            ${CMAKE_CURRENT_SOURCE_DIR}/host/msg
                            ${CMAKE_CURRENT_SOURCE_DIR}/host/util)
target_link_directories(${PIM_TENSOR_STORE_LIB_NAME} PUBLIC ${UPMEM_SDK_DIR}/lib)
target_link_libraries(${PIM_TENSOR_STORE_LIB_NAME} dpu dpuverbose)

install(TARGETS ${PIM_TENSOR_STORE_LIB_NAME} LIBRARY DESTINATION lib)