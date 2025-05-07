add_executable(fire-red-asr-onnx fire-red-asr-onnx.cc features.cc)
target_link_libraries(
    fire-red-asr-onnx 
    onnxruntime 
    gflags::gflags
    kaldi-native-fbank-core
)

add_executable(gflags_test gflags_test.cc)
target_link_libraries(gflags_test gflags::gflags)