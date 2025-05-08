# include_directories(${PROJECT_SOURCE_DIR})
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(
  sources
  features.cc
  whisper-features.cc
  offline-whisper-greedy-search-decoder.cc
  offline-whisper-model.cc
  offline-whisper-model-config.cc
  onnx-utils.cc
  text-utils.cc
  wave-reader.cc
  base64-decode.cc
  bbpe.cc
  symbol-table.cc
)

add_library(recognizer SHARED ${sources})
target_include_directories(recognizer PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(
  recognizer
  onnxruntime
  kaldi-native-fbank-core
)

install(TARGETS recognizer DESTINATION lib)