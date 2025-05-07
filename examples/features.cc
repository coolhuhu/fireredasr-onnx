/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "features.h"

#include <algorithm>
#include <memory>
#include <mutex>  // NOLINT
#include <vector>
#include <iostream>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace sherpa_ncnn {

std::string FeatureExtractorConfig::ToString() const {
  std::ostringstream os;

  os << "FeatureExtractorConfig(";
  os << "sampling_rate=" << sampling_rate << ", ";
  os << "feature_dim=" << feature_dim << ")";

  return os.str();
}

class FeatureExtractor::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config) {
    opts_.frame_opts.dither = 0;
    opts_.frame_opts.snip_edges = false;
    opts_.frame_opts.samp_freq = config.sampling_rate;

    opts_.mel_opts.num_bins = config.feature_dim;

    // Please see
    // https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/fbank.py#L27
    // and
    // https://github.com/k2-fsa/sherpa-onnx/issues/514
    opts_.mel_opts.high_freq = -400;

    fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
  }

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    fbank_->AcceptWaveform(sampling_rate, waveform, n);
  }

  void InputFinished() {
    std::lock_guard<std::mutex> lock(mutex_);
    fbank_->InputFinished();
  }

  int32_t NumFramesReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fbank_->NumFramesReady();
  }

  bool IsLastFrame(int32_t frame) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fbank_->IsLastFrame(frame);
  }

  std::vector<float>
    GetFrames(int32_t frame_index, int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame_index + n > fbank_->NumFramesReady()) {
      std::cerr << frame_index << " + " << n << " > "
                << fbank_->NumFramesReady() << "\n";
      exit(-1);
    }

    int32_t discard_num = frame_index - last_frame_index_;
    if (discard_num < 0) {
      std::cerr << "last_frame_index_: " << last_frame_index_ << ", frame_index_: "
                << frame_index << "\n";
      exit(-1);
    }

    fbank_->Pop(discard_num);

    int32_t feature_dim = fbank_->Dim();
    std::vector<float> features(n * feature_dim);
    float *p = features.data();
    for (int32_t i = 0; i != n; ++i) {
      const float *f = fbank_->GetFrame(i + frame_index);
      std::copy(f, f + feature_dim, p);
      p += feature_dim;
    }

    last_frame_index_ = frame_index;

    return features;
  }

 private:
  std::unique_ptr<knf::OnlineFbank> fbank_;
  knf::FbankOptions opts_;
  mutable std::mutex mutex_;
  int32_t last_frame_index_ = 0;
};

FeatureExtractor::FeatureExtractor(const FeatureExtractorConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FeatureExtractor::~FeatureExtractor() = default;

void FeatureExtractor::AcceptWaveform(int32_t sampling_rate,
                                      const float *waveform, int32_t n) {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

void FeatureExtractor::InputFinished() { impl_->InputFinished(); }

int32_t FeatureExtractor::NumFramesReady() const {
  return impl_->NumFramesReady();
}

bool FeatureExtractor::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

std::vector<float>
  FeatureExtractor::GetFrames(int32_t frame_index, int32_t n) const {
  return impl_->GetFrames(frame_index, n);
}

}  // namespace sherpa_ncnn
