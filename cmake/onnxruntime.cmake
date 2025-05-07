
message("download onnxruntime...")
include(FetchContent)

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

set(ONNXRUNTIME_VERSION 1.21.0)
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
    include(onnxruntime-osx-arm64)
  else()
    message("unsupported ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR} platform")
  endif()
else()
  message("unsupported ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR} platform")
endif()
