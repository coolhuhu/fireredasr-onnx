#include "onnxruntime_cxx_api.h"
#include "gflags/gflags.h"
#include "recognizer/wav.h"
#include "recognizer/features.h"

#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <random>
#include <array>


DEFINE_string(encoder_file, "", "Path to encoder model file");
DEFINE_string(decoder_file, "", "Path to decoder model file");
DEFINE_string(wav_file, "", "Path to wav file");

std::string print_shape(const std::vector<std::int64_t>& v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

std::vector<float> generate_random_input(int input_len) {
  std::random_device rd;
  std::mt19937 gen(rd());  // Mersenne Twister 伪随机数生成器
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f); // 范围 [0.0, 1.0)

  std::vector<float> input(input_len);
  std::generate(input.begin(), input.end(), [&]() { return dis(gen); });
  return input;
}

static std::string GetInputName(Ort::Session *sess, size_t index,
  OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = sess->GetInputNameAllocated(index, allocator);
  return v.get();
#else
  auto v = sess->GetInputName(index, allocator);
  std::string ans = v;
  allocator->Free(allocator, v);
  return ans;
#endif
}

static std::string GetOutputName(Ort::Session *sess, size_t index,
   OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = sess->GetOutputNameAllocated(index, allocator);
  return v.get();
#else
  auto v = sess->GetOutputName(index, allocator);
  std::string ans = v;
  allocator->Free(allocator, v);
  return ans;
#endif
}

void GetModelInputNames(Ort::Session *sess, 
                        std::vector<std::string> *input_names,
                        std::vector<const char *> *input_names_ptr) {
                          Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetInputCount();
  input_names->resize(node_count);
  input_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*input_names)[i] = GetInputName(sess, i, allocator);
    (*input_names_ptr)[i] = (*input_names)[i].c_str();
  }
}

void GetOutputNames(Ort::Session *sess, 
                    std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetOutputCount();
  output_names->resize(node_count);
  output_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*output_names)[i] = GetOutputName(sess, i, allocator);
    (*output_names_ptr)[i] = (*output_names)[i].c_str();
  }
}


class FireRedASRAEDModel {

private:
  
};


std::vector<float> ReadWav(const std::string& wav_path) {
  wenet::WavReader wav_reader(wav_path);

  std::cout << "num channel: " << wav_reader.num_channel() << std::endl;
  std::cout << "sample rate: " << wav_reader.sample_rate() << std::endl;
  std::cout << "bits per sample: " << wav_reader.bits_per_sample() << std::endl;
  std::cout << "num samples: " << wav_reader.num_samples() << std::endl;

  sherpa_ncnn::FeatureExtractorConfig config;
  config.sampling_rate = wav_reader.sample_rate();
  sherpa_ncnn::FeatureExtractor feature_extractor(config);

  feature_extractor.AcceptWaveform(wav_reader.sample_rate(), 
                                   wav_reader.data(), 
                                   wav_reader.num_samples());
  feature_extractor.InputFinished();
  
  std::cout << feature_extractor.NumFramesReady() << std::endl;

  auto fbank_features = feature_extractor.GetFrames(0, feature_extractor.NumFramesReady());

  std::cout << "fbank_features.size() = " << fbank_features.size() << std::endl;

  for (int i = 0; i < 10; ++i) {
    std::cout << fbank_features[i] << " ";
  }
  std::cout << std::endl;

  return fbank_features;
}


int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  std::cout << "encoder file: " << FLAGS_encoder_file << "\n"
            << "decoder file: " << FLAGS_decoder_file << "\n"
            << "wav file: " << FLAGS_wav_file << "\n";

  std::vector<float> features = ReadWav(FLAGS_wav_file);
  

  std::shared_ptr<Ort::Session> encoder_session = nullptr;
  std::shared_ptr<Ort::Session> decoder_session = nullptr;

  Ort::Env env;
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
    OrtDeviceAllocator, OrtMemTypeDefault
  );

  encoder_session = std::make_shared<Ort::Session>(
    env, FLAGS_encoder_file.c_str(), session_options
  );
  decoder_session = std::make_shared<Ort::Session>(
    env, FLAGS_decoder_file.c_str(), session_options
  );

  int feat_dim = 80;
  std::int64_t num_frames = features.size() / feat_dim;
  std::array<std::int64_t, 3> input_shape = {1, num_frames, feat_dim};
  
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, features.data(), features.size(), 
    input_shape.data(), input_shape.size()
  );

  std::int64_t input_len_shape = 1;
  Ort::Value input_len_tensor = Ort::Value::CreateTensor(
    memory_info, &num_frames, 1, 
    &input_len_shape, 1
  );

  std::vector<std::string> encoder_input_names;
  std::vector<const char *> encoder_input_names_ptr;
  GetModelInputNames(encoder_session.get(), &encoder_input_names, &encoder_input_names_ptr);

  std::vector<std::string> encoder_output_names;
  std::vector<const char *> encoder_output_names_ptr;
  GetOutputNames(encoder_session.get(), &encoder_output_names, &encoder_output_names_ptr);
  
  std::array<Ort::Value, 2> encoder_inputs{
    std::move(input_tensor),
    std::move(input_len_tensor)
  };
  auto encoder_outputs = encoder_session->Run(
    {},
    encoder_input_names_ptr.data(), encoder_inputs.data(), encoder_inputs.size(),
    encoder_output_names_ptr.data(), encoder_output_names_ptr.size()
  );
  std::cout << "encoder outputs size: " << encoder_outputs.size() << "\n";

  Ort::Value n_layer_cross_k = std::move(encoder_outputs[0]);
  Ort::Value n_layer_cross_v = std::move(encoder_outputs[1]);
  Ort::Value cross_attn_mask = std::move(encoder_outputs[2]);
  
  std::int64_t sos_id = 3, eos_id = 4;

  // init self kv attention cache
  Ort::AllocatorWithDefaultOptions allocator;
  std::int64_t num_layer = 16, batch_size = 1, max_decode_len = 448, hidden_dim = 1280;
  std::array<std::int64_t, 4> self_kv_cache_shape = {
    num_layer, batch_size, max_decode_len, hidden_dim
  };
  Ort::Value n_layer_self_k_cache = Ort::Value::CreateTensor<float>(
    allocator, self_kv_cache_shape.data(), self_kv_cache_shape.size()
  );
  Ort::Value n_layer_self_v_cache = Ort::Value::CreateTensor<float>(
    allocator, self_kv_cache_shape.data(), self_kv_cache_shape.size()

  );
  auto n = num_layer * batch_size * max_decode_len * hidden_dim;
  float *p_k = n_layer_self_k_cache.GetTensorMutableData<float>();
  float *p_v = n_layer_self_v_cache.GetTensorMutableData<float>();
  memset(p_k, 0, sizeof(float) * n);
  memset(p_v, 0, sizeof(float) * n);

  // (batch_size, n_tokens)
  std::array<std::int64_t, 2> token_shape = {1, 1};
  std::int64_t token = sos_id;

  std::int64_t offset = 0;
  std::array<std::int64_t, 1> offset_shape{1};
  Ort::Value offset_tensor = Ort::Value::CreateTensor(
    memory_info, &offset, 1, offset_shape.data(), offset_shape.size());

  std::vector<std::string> decoder_input_names;
  std::vector<const char *> decoder_input_names_ptr;
  GetModelInputNames(decoder_session.get(), &decoder_input_names, &decoder_input_names_ptr);

  std::vector<std::string> decoder_output_names;
  std::vector<const char *> decoder_output_names_ptr;
  GetOutputNames(decoder_session.get(), &decoder_output_names, &decoder_output_names_ptr);


  std::vector<std::int64_t> decode_results;
  for (int i = 0; i < 448; ++i) {
    // Autoregressive Decoding
    Ort::Value token_tensor = Ort::Value::CreateTensor(
      memory_info, &token, 1, token_shape.data(), token_shape.size());

    std::array<Ort::Value, 7> decoder_inputs{
      std::move(token_tensor),
      std::move(n_layer_self_k_cache),
      std::move(n_layer_self_v_cache),
      std::move(n_layer_cross_k),
      std::move(n_layer_cross_v),
      std::move(cross_attn_mask),
      std::move(offset_tensor)
    };

    auto decoder_outputs = decoder_session->Run(
      {},
      decoder_input_names_ptr.data(),
      decoder_inputs.data(),
      decoder_inputs.size(),
      decoder_output_names_ptr.data(),
      decoder_output_names_ptr.size()
    );

    Ort::Value logits = std::move(decoder_outputs[0]);
    n_layer_self_k_cache = std::move(decoder_outputs[1]);
    n_layer_self_v_cache = std::move(decoder_outputs[2]);
    n_layer_cross_k = std::move(decoder_outputs[3]);
    n_layer_cross_v = std::move(decoder_outputs[4]);
    cross_attn_mask = std::move(decoder_outputs[5]);
    offset_tensor = std::move(decoder_outputs[6]);


    const float *p_logits = logits.GetTensorData<float>();

    auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    int32_t vocab_size = logits_shape[2];
    int32_t max_token_id = static_cast<int32_t>(std::distance(
      p_logits, std::max_element(p_logits, p_logits + vocab_size)));

    if (max_token_id == eos_id) {
      break;
    }

    token = max_token_id;
    *(offset_tensor.GetTensorMutableData<std::int64_t>()) += 1;

    decode_results.push_back(token);

    std::cout << token << " ";
  }
  std::cout << std::endl;

  std::cout << "Decoded done." << std::endl;
  std::cout << "Decoded results: " << std::endl;
  for (const auto& res : decode_results) {
    std::cout << res << " ";
  }
  std::cout << std::endl;

}