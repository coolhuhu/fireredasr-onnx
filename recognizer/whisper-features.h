#ifndef WHISPER_FEATURES_H_
#define WHISPER_FEATURES_H_

#include <memory>
#include <vector>

struct WhisperFeatureConfig {
  int sample_rate = 16000;
  int feature_dim = 80;
};


class WhisperFeatureExtractor {
public:
  explicit WhisperFeatureExtractor(const WhisperFeatureConfig& config);

  ~WhisperFeatureExtractor();

  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const;

  int32_t FeatureDim() const;

  std::vector<float> GetFrames() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};


#endif