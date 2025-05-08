#ifndef OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_
#define OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_

#include "offline-whisper-model-config.h"
#include "offline-whisper-model.h"

namespace sherpa_onnx {

struct OfflineWhisperDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;
  std::string lang;
};

class OfflineWhisperGreedySearchDecoder {
public:
  OfflineWhisperGreedySearchDecoder(const OfflineWhisperModelConfig &config,
                                    OfflineWhisperModel *model)
      : config_(config), model_(model) {}

  std::vector<OfflineWhisperDecoderResult> Decode(
      Ort::Value cross_k, Ort::Value cross_v,
      int32_t num_feature_frames);

  void SetConfig(const OfflineWhisperModelConfig &config);

private:
  OfflineWhisperModelConfig config_;
  OfflineWhisperModel *model_;  // not owned
};

} // namespace sherpa_onnx

#endif