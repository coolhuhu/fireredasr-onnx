if(NOT CMAKE_SYSTEM_NAME STREQUAL Darwin)
  message(FATAL_ERROR "This file is for macOS only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if (NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
  message(FATAL_ERROR "This file is for macOS arm64 only. Given: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")
set(URL_HASH "SHA256=5c3f2064ee97eb7774e87f396735c8eada7287734f1bb7847467ad30d4036115")

set(possible_file_locations "${CMAKE_SOURCE_DIR}/third-part/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")

if (EXISTS ${possible_file_locations})
  set(URL "${possible_file_locations}")
  file(TO_CMAKE_PATH "${URL}" URL)
  message(STATUS "Found local download onnxruntime: ${URL}")
endif()

FetchContent_Declare(
  onnxruntime
  URL ${URL}
  URL_HASH ${URL_HASH}
)
FetchContent_MakeAvailable(onnxruntime)

find_library(location_onnxruntime onnxruntime
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
message(STATUS "onnxruntime library locate at ${location_onnxruntime}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${location_onnxruntime}"
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)




