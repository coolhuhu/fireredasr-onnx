#include "offline-whisper-model-config.h"

#include <sstream>


namespace sherpa_onnx {

	std::string OfflineWhisperModelConfig::ToString() const {
		std::ostringstream os;
	
		os << "OfflineWhisperModelConfig(";
		os << "encoder=\"" << encoder << "\", ";
		os << "decoder=\"" << decoder << "\", ";
		os << "language=\"" << language << "\", ";
		os << "task=\"" << task << "\", ";
		os << "provider=\"" << provider << "\", ";
		os << "tail_paddings=" << tail_paddings << "\", ";
		os << "inter_op_num_threads=" << inter_op_num_threads << "\", ";
		os << "intra_op_num_threads=" << intra_op_num_threads << ")";
	
		return os.str();
	}

} // namespace sherpa-onnx