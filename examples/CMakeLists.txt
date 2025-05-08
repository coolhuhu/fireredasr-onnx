# include_directories(${CMAKE_SOURCE_DIR})

add_executable(fire-red-asr-onnx fire-red-asr-onnx.cc)
target_include_directories(
  fire-red-asr-onnx
  PUBLIC
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(
  fire-red-asr-onnx 
  recognizer
  gflags::gflags
)

add_executable(gflags_test gflags_test.cc)
target_link_libraries(gflags_test gflags::gflags)

add_executable(whisper-cxx-api whisper-cxx-api.cc)
target_include_directories(
  whisper-cxx-api
  PUBLIC
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(
  whisper-cxx-api
  recognizer
  gflags::gflags
)