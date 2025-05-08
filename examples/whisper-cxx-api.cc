#include "gflags/gflags.h"
#include "recognizer/offline-whisper-model-config.h"
#include "recognizer/offline-whisper-model.h"
#include "recognizer/offline-whisper-greedy-search-decoder.h"
#include "recognizer/wav.h"
#include "recognizer/features.h"
#include "recognizer/whisper-features.h"
#include "recognizer/wave-reader.h"

#include "kaldi-native-fbank/csrc/online-feature.h"

#include <iostream>
#include <memory>

DEFINE_string(encoder_file, "", "Path to encoder model file");
DEFINE_string(decoder_file, "", "Path to decoder model file");
DEFINE_string(language, "en", "Path to decoder model file");
DEFINE_string(wav_file, "", "Path to wav file");


template <typename T /*= float*/>
Ort::Value Transpose12(OrtAllocator *allocator, const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  assert(shape.size() == 3);

  std::array<int64_t, 3> ans_shape{shape[0], shape[2], shape[1]};
  Ort::Value ans = Ort::Value::CreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
  T *dst = ans.GetTensorMutableData<T>();
  auto row_stride = shape[2];
  for (int64_t b = 0; b != ans_shape[0]; ++b) {
    const T *src = v->GetTensorData<T>() + b * shape[1] * shape[2];
    for (int64_t i = 0; i != ans_shape[1]; ++i) {
      for (int64_t k = 0; k != ans_shape[2]; ++k, ++dst) {
        *dst = (src + k * row_stride)[i];
      }
    }
  }

  return ans;
}



int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  sherpa_onnx::OfflineWhisperModelConfig config;
  config.encoder = FLAGS_encoder_file;
  config.decoder = FLAGS_decoder_file;
  // config.language = FLAGS_language;
  std::cout << config.ToString() << std::endl;

  // 1. load model
  std::unique_ptr<sherpa_onnx::OfflineWhisperModel> model(
      new sherpa_onnx::OfflineWhisperModel(config));
  std::unique_ptr<sherpa_onnx::OfflineWhisperGreedySearchDecoder> decoder(
      new sherpa_onnx::OfflineWhisperGreedySearchDecoder(config, model.get()));
  decoder->SetConfig(config);

  int32_t max_num_frames = 3000;
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // 2. load wav and extractor features.
  //    
  // FIXME: whisper-large-v3 feat dim is 128

  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> waveform =sherpa_onnx::ReadWave(FLAGS_wav_file, &sample_rate, &is_ok);
  if (!is_ok) {
    std::cerr << "read wav " << FLAGS_wav_file << " error." << std::endl;
    exit(-1);
  }

  WhisperFeatureConfig whisper_feature_config;
  whisper_feature_config.sample_rate = 16000;
  whisper_feature_config.feature_dim = 80;

  WhisperFeatureExtractor whipser_feature_extractor(whisper_feature_config);
  whipser_feature_extractor.AcceptWaveform(sample_rate, waveform.data(), 
                                           waveform.size());
  
  int feature_dim = whipser_feature_extractor.FeatureDim();
  std::vector<float> features = whipser_feature_extractor.GetFrames();

  int32_t num_frames = features.size() / feature_dim;

  // we use 50 here so that there will be some zero tail paddings
  if (num_frames >= max_num_frames - 50) {
    std::cerr <<
        "Only waves less than 30 seconds are supported. We process only the "
        "first 30 seconds and discard the remaining data";
    num_frames = max_num_frames - 50;
  }

  model->NormalizeFeatures(features.data(), num_frames, feature_dim);

  // note that 1000 is an experience-value.
  // You can replace 1000 by other values, say, 100.
  //
  // Since we have removed the 30 seconds constraint, we need
  // tail_padding_frames so that whisper is able to detect the eot token.
  int32_t tail_padding_frames = 100;

  if (config.tail_paddings > 0) {
    tail_padding_frames = config.tail_paddings;
  }

  int32_t actual_frames =
      std::min(num_frames + tail_padding_frames, max_num_frames);

  std::array<int64_t, 3> shape{1, actual_frames, feature_dim};

  Ort::Value mel = Ort::Value::CreateTensor<float>(
      model->Allocator(), shape.data(), shape.size());

  float *p_mel = mel.GetTensorMutableData<float>();
  std::copy(features.data(), features.data() + num_frames * feature_dim, p_mel);

  std::fill_n(p_mel + num_frames * feature_dim,
              (actual_frames - num_frames) * feature_dim, 0);

  mel = Transpose12<float>(model->Allocator(), &mel);

  p_mel = mel.GetTensorMutableData<float>();
  
  // 3. inference.
  auto cross_kv = model->ForwardEncoder(std::move(mel));
  auto results = decoder->Decode(std::move(cross_kv.first),
                                 std::move(cross_kv.second), num_frames);
  
  std::cout << "lang: " << results[0].lang << std::endl;
  for (const auto& t : results[0].tokens) {
    std::cout << t << " ";
  }
  std::cout << std::endl;
}