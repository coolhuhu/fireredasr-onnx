cmake_minimum_required(VERSION 3.16)
project(FireRedASRONNX)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(onnxruntime)
include(gflags)
include(kaldi-native-fbank)


add_subdirectory(examples)
add_subdirectory(recognizer)