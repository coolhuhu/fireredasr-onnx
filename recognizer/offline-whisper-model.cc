#include "offline-whisper-model.h"
#include "macros.h"
#include "onnx-utils.h"

#include <fstream>
#include <sstream>


namespace sherpa_onnx {

std::vector<char> ReadFile(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  return buffer;
}


class OfflineWhisperModel::Impl {
public:
  explicit Impl(const OfflineWhisperModelConfig &config)
    : config_(config),
      env_(ORT_LOGGING_LEVEL_ERROR)
      sess_opts_(sherpa_onnx::GetSessionOptions(
        config.inter_op_num_threads,
        config.intra_op_num_threads,
        config.provider
      )),
      allocator_{} {

    {
      auto buf = ReadFile(config.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  std::pair<Ort::Value, Ort::Value> ForwardEncoder(Ort::Value features) {
    auto encoder_out = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), &features, 1,
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());

    return {std::move(encoder_out[0]), std::move(encoder_out[1])};
  }

  std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
             Ort::Value>
  ForwardDecoder(Ort::Value tokens, Ort::Value n_layer_self_k_cache,
                 Ort::Value n_layer_self_v_cache, Ort::Value n_layer_cross_k,
                 Ort::Value n_layer_cross_v, Ort::Value offset) {
    std::array<Ort::Value, 6> decoder_input = {std::move(tokens),
                                               std::move(n_layer_self_k_cache),
                                               std::move(n_layer_self_v_cache),
                                               std::move(n_layer_cross_k),
                                               std::move(n_layer_cross_v),
                                               std::move(offset)};

    auto decoder_out = decoder_sess_->Run(
    {}, decoder_input_names_ptr_.data(), decoder_input.data(),
    decoder_input.size(), decoder_output_names_ptr_.data(),
    decoder_output_names_ptr_.size());

    return std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value,
                      Ort::Value, Ort::Value> {
      std::move(decoder_out[0]),   std::move(decoder_out[1]),
      std::move(decoder_out[2]),   std::move(decoder_input[3]),
      std::move(decoder_input[4]), std::move(decoder_input[5])};
  }

  std::pair<Ort::Value, Ort::Value> GetInitialSelfKVCache() {
  }

  int32_t DetectLanguage(Ort::Value &cross_k,    // NOLINT
                         Ort::Value &cross_v) {  // NOLINT
    int64_t token_val = SOT();
    std::array<int64_t, 2> token_shape{1, 1};

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value tokens = Ort::Value::CreateTensor(
        memory_info, &token_val, 1, token_shape.data(), token_shape.size());

    auto self_kv_cache = GetInitialSelfKVCache();

    std::array<int64_t, 1> offset_shape{1};
    Ort::Value offset = Ort::Value::CreateTensor<int64_t>(
                                    Allocator(), offset_shape.data(), offset_shape.size());
    *(offset.GetTensorMutableData<int64_t>()) = 0;

    auto decoder_out =
        ForwardDecoder(std::move(tokens), std::move(self_kv_cache.first),
                        std::move(self_kv_cache.second), std::move(cross_k),
                        std::move(cross_v), std::move(offset));

    cross_k = std::move(std::get<3>(decoder_out));
    cross_v = std::move(std::get<4>(decoder_out));

    const float *p_logits = std::get<0>(decoder_out).GetTensorData<float>();
    const auto &all_language_ids = GetAllLanguageIDs();

    int32_t lang_id = all_language_ids[0];
    float this_logit = p_logits[lang_id];

    for (int32_t i = 1; i != all_language_ids.size(); ++i) {
      int32_t id = all_language_ids[i];
      float p = p_logits[id];

      if (p > this_logit) {
        this_logit = p;
        lang_id = id;
      }
    }

    return lang_id;
  }

  OrtAllocator *Allocator() { return allocator_; }

  const std::vector<int64_t> &GetInitialTokens() const { return sot_sequence_; }

  const std::vector<int32_t> &GetAllLanguageIDs() const {
    return all_language_tokens_;
  }

  const std::unordered_map<std::string, int32_t> &GetLang2ID() const {
    return lang2id_;
  }

  const std::unordered_map<int32_t, std::string> &GetID2Lang() const {
    return id2lang_;
  }

  int32_t NoTimeStampsToken() const { return no_timestamps_; }

  int32_t EOT() const { return eot_; }

  int32_t SOT() const { return sot_; }

  int32_t TextCtx() const { return n_text_ctx_; }

  int32_t VocabSize() const { return n_vocab_; }

  int32_t FeatureDim() const { return n_mels_; }

  int32_t Translate() const { return translate_; }

  bool IsMultiLingual() const { return is_multilingual_; }

private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    encoder_sess_ = std::make_unique<Ort::Session>(
      env_, model_data, model_data_length, sess_opts_);

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                  &encoder_output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
    
    std::ostringstream os;
    os << "---encoder---\n";
    PrintModelMetadata(os, meta_data);

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(n_mels_, "n_mels");
    SHERPA_ONNX_READ_META_DATA(n_text_layer_, "n_text_layer");
    SHERPA_ONNX_READ_META_DATA(n_text_ctx_, "n_text_ctx");
    SHERPA_ONNX_READ_META_DATA(n_text_state_, "n_text_state");
    SHERPA_ONNX_READ_META_DATA(n_vocab_, "n_vocab");
    SHERPA_ONNX_READ_META_DATA(sot_, "sot");
    SHERPA_ONNX_READ_META_DATA(eot_, "eot");
    SHERPA_ONNX_READ_META_DATA(blank_, "blank_id");
    SHERPA_ONNX_READ_META_DATA(translate_, "translate");
    SHERPA_ONNX_READ_META_DATA(transcribe_, "transcribe");
    SHERPA_ONNX_READ_META_DATA(is_multilingual_, "is_multilingual");
    SHERPA_ONNX_READ_META_DATA(no_timestamps_, "no_timestamps");
    SHERPA_ONNX_READ_META_DATA(no_speech_, "no_speech");
    SHERPA_ONNX_READ_META_DATA_VEC(sot_sequence_, "sot_sequence");

    if (is_multilingual_) {
      SHERPA_ONNX_READ_META_DATA_VEC(all_language_tokens_,
                                    "all_language_tokens");
      SHERPA_ONNX_READ_META_DATA_VEC_STRING(all_language_codes_,
                                            "all_language_codes");
      if (all_language_tokens_.size() != all_language_codes_.size()) {
        SHERPA_ONNX_LOGE("# lang_id: %d != # lang_code: %d",
                        static_cast<int32_t>(all_language_tokens_.size()),
                        static_cast<int32_t>(all_language_codes_.size()));
        exit(-1);
      }

      for (int32_t i = 0;
          i != static_cast<int32_t>(all_language_tokens_.size()); ++i) {
        lang2id_[all_language_codes_[i]] = all_language_tokens_[i];
        id2lang_[all_language_tokens_[i]] = all_language_codes_[i];
      }
    }
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    decoder_sess_ = std::make_unique<Ort::Session>(
      env_, model_data, model_data_length, sess_opts_);

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                  &decoder_output_names_ptr_);
  }


private:
  OfflineOfflineWhisperModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<int32_t> all_language_tokens_;
  std::vector<std::string> all_language_codes_;
  std::unordered_map<std::string, int32_t> lang2id_;
  std::unordered_map<int32_t, std::string> id2lang_;

  // model meta data
  int32_t n_mels_ = 80;
  int32_t n_text_layer_ = 0;
  int32_t n_text_ctx_ = 0;
  int32_t n_text_state_ = 0;
  int32_t n_vocab_ = 0;
  int32_t sot_ = 0;
  int32_t eot_ = 0;
  int32_t blank_ = 0;
  int32_t translate_ = 0;
  int32_t transcribe_ = 0;
  int32_t no_timestamps_ = 0;
  int32_t no_speech_ = 0;
  int32_t is_multilingual_ = 0;
  std::vector<int64_t> sot_sequence_;
};


OfflineWhisperModel::~OfflineWhisperModel() = default;

std::pair<Ort::Value, Ort::Value> OfflineWhisperModel::ForwardEncoder(
    Ort::Value features) const {
  return impl_->ForwardEncoder(std::move(features));
}

std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
           Ort::Value>
OfflineWhisperModel::ForwardDecoder(Ort::Value tokens,
                                    Ort::Value n_layer_self_k_cache,
                                    Ort::Value n_layer_self_v_cache,
                                    Ort::Value n_layer_cross_k,
                                    Ort::Value n_layer_cross_v,
                                    Ort::Value offset) const {
  return impl_->ForwardDecoder(
      std::move(tokens), std::move(n_layer_self_k_cache),
      std::move(n_layer_self_v_cache), std::move(n_layer_cross_k),
      std::move(n_layer_cross_v), std::move(offset));
}

int32_t OfflineWhisperModel::DetectLanguage(Ort::Value &cross_k,    // NOLINT
                                            Ort::Value &cross_v) {  // NOLINT
  return impl_->DetectLanguage(cross_k, cross_v);
}

std::pair<Ort::Value, Ort::Value> OfflineWhisperModel::GetInitialSelfKVCache()
    const {
  return impl_->GetInitialSelfKVCache();
}

OrtAllocator *OfflineWhisperModel::Allocator() const {
  return impl_->Allocator();
}

const std::vector<int64_t> &OfflineWhisperModel::GetInitialTokens() const {
  return impl_->GetInitialTokens();
}

const std::vector<int32_t> &OfflineWhisperModel::GetAllLanguageIDs() const {
  return impl_->GetAllLanguageIDs();
}

const std::unordered_map<std::string, int32_t>
    &OfflineWhisperModel::GetLang2ID() const {
  return impl_->GetLang2ID();
}

const std::unordered_map<int32_t, std::string>
    &OfflineWhisperModel::GetID2Lang() const {
  return impl_->GetID2Lang();
}

int32_t OfflineWhisperModel::NoTimeStampsToken() const {
  return impl_->NoTimeStampsToken();
}

int32_t OfflineWhisperModel::EOT() const { return impl_->EOT(); }

int32_t OfflineWhisperModel::SOT() const { return impl_->SOT(); }

int32_t OfflineWhisperModel::TextCtx() const { return impl_->TextCtx(); }

int32_t OfflineWhisperModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OfflineWhisperModel::FeatureDim() const { return impl_->FeatureDim(); }

int32_t OfflineWhisperModel::Translate() const { return impl_->Translate(); }

bool OfflineWhisperModel::IsMultiLingual() const {
  return impl_->IsMultiLingual();
}

void OfflineWhisperModel::NormalizeFeatures(float *features, int32_t num_frames,
                                            int32_t feat_dim) {
  // log_spec = torch.clamp(features, min=1e-10).log10()
  // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
  // mel = (log_spec + 4.0) / 4.0

  int32_t n = num_frames * feat_dim;
  float max_v = -1e20;
  for (int32_t i = 0; i != n; ++i) {
    float f = features[i];

    f = std::max<float>(f, 1e-10);
    f = std::log10(f);

    max_v = std::max(f, max_v);

    features[i] = f;
  }

  max_v -= 8;

  for (int32_t i = 0; i != n; ++i) {
    float f = features[i];
    f = std::max(f, max_v);

    f = (f + 4) / 4;

    features[i] = f;
  }
}

} // namespace sherpa_onnx