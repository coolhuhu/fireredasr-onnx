#ifndef OFFLINE_WHISPER_MODEL_CONFIG_H_
#define OFFLINE_WHISPER_MODEL_CONFIG_H_

#include <string>

namespace sherpa_onnx {

struct OfflineWhisperModelConfig {
  std::string encoder;
  std::string decoder;

  // Available languages can be found at
  // https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
  //
  // Note: For non-multilingual models, it supports only "en"
  //
  // If empty, we will infer it from the input audio file when
  // the model is multilingual.
  std::string language;

  // Valid values are transcribe and translate
  //
  // Note: For non-multilingual models, it supports only "transcribe"
  std::string task = "transcribe";

  std::string provider = "cpu";

  // Number of tail padding frames.
  //
  // Since we remove the 30-second constraint, we need to add some paddings
  // at the end.
  //
  // Recommended values:
  //   - 50 for English models
  //   - 300 for multilingual models
  int32_t tail_paddings = -1;

  int inter_op_num_threads = 1;

  int intra_op_num_threads = 1;

  OfflineWhisperModelConfig() = default;

  OfflineWhisperModelConfig(const OfflineWhisperModelConfig& config)
    : encoder(config.encoder),
      decoder(config.decoder),
      language(config.language),
      task(config.task),
      provider(config.provider),
      tail_paddings(config.tail_paddings),
      inter_op_num_threads(config.inter_op_num_threads),
      intra_op_num_threads(config.intra_op_num_threads) {}

  OfflineWhisperModelConfig(const std::string &encoder,
                            const std::string &decoder,
                            const std::string &language,
                            const std::string &task, 
                            const std::string &provider,
                            int32_t tail_paddings,
                            int inter_op_num_threads = 1,
                            int intra_op_num_threads = 1)
    : encoder(encoder),
      decoder(decoder),
      language(language),
      task(task),
      provider(provider),
      tail_paddings(tail_paddings),
      inter_op_num_threads(inter_op_num_threads),
      intra_op_num_threads(intra_op_num_threads) {}
};

// namespace sherpa_onnx

#endif  // OFFLINE_WHISPER_MODEL_CONFIG_H_