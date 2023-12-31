cmake_minimum_required(VERSION 3.9)
cmake_policy(SET CMP0042 NEW)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)

if (DEFINED ENV{LIBTORCH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH}")
endif()

include(FetchContent)

project (type-torch)

# Set remote URL for libtorch
set (LIBTORCH_VERSION 2.0.1)
set (LIBTORCH_COMPUTE_PLATFORM cu118)
FetchContent_Declare(
        Torch
        URL https://download.pytorch.org/libtorch/${LIBTORCH_COMPUTE_PLATFORM}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_COMPUTE_PLATFORM}.zip
        FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(Torch)

include_directories(${CMAKE_JS_INC})
file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/csrc/*.cc" "${PROJECT_SOURCE_DIR}/csrc/*.h")
add_library(${PROJECT_NAME} SHARED ${SRC_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

# Include Node-API wrappers
execute_process(COMMAND node -p "require('node-addon-api').include"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NODE_ADDON_API_DIR
        )
string(REGEX REPLACE "[\r\n\"]" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})


target_include_directories(${PROJECT_NAME} PRIVATE ${NODE_ADDON_API_DIR})

# define NAPI_VERSION
add_definitions(-DNAPI_VERSION=3)
