#include "whisper-features.h"

#include "kaldi-native-fbank/csrc/online-feature.h"

class WhisperFeatureExtractor::Impl {
public:
  explicit Impl(const WhisperFeatureConfig& config)
      : config_(config) {
    opts_.frame_opts.samp_freq = config.sample_rate;
    opts_.mel_opts.num_bins = config.feature_dim;

    knf::WhisperFeatureOptions whisper_opts;
    whisper_opts.frame_opts = opts_.frame_opts;
    whisper_opts.dim = config.feature_dim;

    whisper_fbank_ = std::make_unique<knf::OnlineWhisperFbank>(whisper_opts);
  }

  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
      int32_t n) const {
    whisper_fbank_->AcceptWaveform(sampling_rate, waveform, n);
    whisper_fbank_->InputFinished();
  }

  int32_t FeatureDim() const {
    return opts_.mel_opts.num_bins;
  }

  std::vector<float> GetFrames() const {
    int32_t n = whisper_fbank_->NumFramesReady();
    int32_t feature_dim = FeatureDim();

    std::vector<float> features(n * feature_dim);

    float *p = features.data();

    for (int32_t i = 0; i != n; ++i) {
      const float *f = whisper_fbank_->GetFrame(i);
      std::copy(f, f + feature_dim, p);
      p += feature_dim;
    }

    return features;
  }

private:
  WhisperFeatureConfig config_;
  std::unique_ptr<knf::OnlineWhisperFbank> whisper_fbank_;
  knf::FbankOptions opts_;
};


WhisperFeatureExtractor::WhisperFeatureExtractor(const WhisperFeatureConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

WhisperFeatureExtractor::~WhisperFeatureExtractor() = default;

void WhisperFeatureExtractor::AcceptWaveform(int32_t sampling_rate, const float *waveform,
                                             int32_t n) const {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

int32_t WhisperFeatureExtractor::FeatureDim() const {
  return impl_->FeatureDim();
}

std::vector<float> WhisperFeatureExtractor::GetFrames() const {
  return impl_->GetFrames();
}