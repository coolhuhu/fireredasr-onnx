#include "gflags/gflags.h"

#include <iostream>


DEFINE_string(encoder_file, "", "onnx encoder file path.");
DEFINE_string(decoder_file, "", "onnx encoder file path.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  std::cout << "encoder_file: " << FLAGS_encoder_file << std::endl;
  std::cout << "decoder_file: " << FLAGS_decoder_file << std::endl;
}