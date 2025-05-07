from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer

import onnxruntime as ort
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Tuple, List
import argparse
import os
import time


# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


class FireRedASROnnxModel:
    def __init__(
        self, 
        encoder_path, 
        decoder_path, 
        providers=["CPUExecutionProvider"]
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        # session_opts.log_severity_level = 1
        self.session_opts = session_opts
        
        # FIXME: 参考whisper设置的最大的解码长度
        # FireRedASR-AED 模型支持的最长语音为 60s
        # ref: https://github.com/FireRedTeam/FireRedASR?tab=readme-ov-file#input-length-limitations
        self.decode_max_len = 448
        
        self.decoder_hidden_dim = 1280
        self.num_decoder_blocks = 16
        self.blank_id = 0
        self.sos_id = 3
        self.eos_id = 4
        self.pad_id = 2
        
        self.encoder = None
        self.decoder = None
        
        self.init_encoder(encoder_path, providers)
        self.init_decoder(decoder_path, providers)
        
    def init_encoder(self, encoder_path, providers=None):
        self.encoder = ort.InferenceSession(
            encoder_path,
            sess_options=self.session_opts,
            providers=providers
        )
    
    def init_decoder(self, decoder_path, providers=None):
        self.decoder = ort.InferenceSession(
            decoder_path,
            sess_options=self.session_opts,
            providers=providers
        )
    
    def run_encoder(self, input: np.ndarray, 
                    input_length: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        n_layer_cross_k, n_layer_cross_v, cross_attn_mask = self.encoder.run(
            None,
            {
                self.encoder.get_inputs()[0].name: input,
                self.encoder.get_inputs()[1].name: input_length
            }
        )
        return (
            n_layer_cross_k,
            n_layer_cross_v,
            cross_attn_mask
        )
        
    def decode_one_token(
        self,
        tokens: np.ndarray,
        n_layer_self_k_cache: np.ndarray,
        n_layer_self_v_cache: np.ndarray,
        n_layer_cross_k_cache: np.ndarray,
        n_layer_cross_v_cache: np.ndarray,
        cross_attn_mask: np.ndarray,
        offset: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache, _, _, _, _ = self.decoder.run(
            None,
            {
                self.decoder.get_inputs()[0].name: tokens,
                self.decoder.get_inputs()[1].name: n_layer_self_k_cache,
                self.decoder.get_inputs()[2].name: n_layer_self_v_cache,
                self.decoder.get_inputs()[3].name: n_layer_cross_k_cache,
                self.decoder.get_inputs()[4].name: n_layer_cross_v_cache,
                self.decoder.get_inputs()[5].name: cross_attn_mask,
                self.decoder.get_inputs()[6].name: offset
            }
        )
        return (
            logits,
            out_n_layer_self_k_cache,
            out_n_layer_self_v_cache
        )
    
    def run_decoder(
        self,
        n_layer_cross_k, 
        n_layer_cross_v, 
        cross_attn_mask
    ) -> List[int]:
        prediction_tokens = torch.tensor([[self.sos_id]], dtype=torch.int64)
        offset = torch.zeros(1, dtype=torch.int64)
        n_layer_self_k_cache, n_layer_self_v_cache = self.get_initialized_self_cache()
        
        results = [self.sos_id]
        for i in range(self.decode_max_len):
            logits, n_layer_self_k_cache, n_layer_self_v_cache = self.decode_one_token(
                to_numpy(prediction_tokens),
                to_numpy(n_layer_self_k_cache),
                to_numpy(n_layer_self_v_cache),
                to_numpy(n_layer_cross_k),
                to_numpy(n_layer_cross_v),
                to_numpy(cross_attn_mask),
                to_numpy(offset)
            )
            
            offset += 1
            logits = torch.from_numpy(logits)
            logits = F.log_softmax(logits.squeeze(0) / 1.25, dim=-1)
            logits = logits.argmax(dim=-1)
            p_token = logits[0].item()
            results.append(p_token)
            prediction_tokens = torch.tensor([[p_token]], dtype=torch.int64)
            if p_token == self.eos_id:
                # EOS
                break
            
        if results[0] == self.sos_id:
            results = results[1:]
        if results[-1] == self.eos_id:
            results = results[:-1]
        return results
        
    def get_initialized_self_cache(self) -> Tuple[Tensor, Tensor]:
        batch_size = 1
        n_layer_self_k_cache = torch.zeros(
            self.num_decoder_blocks,
            batch_size,
            self.decode_max_len,
            self.decoder_hidden_dim,
        )
        n_layer_self_v_cache = torch.zeros(
            self.num_decoder_blocks,
            batch_size,
            self.decode_max_len,
            self.decoder_hidden_dim,
        )
        return n_layer_self_k_cache, n_layer_self_v_cache
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="FireRedASROnnxModel Test")
    parser.add_argument(
        "--encoder", 
        type=str, 
        required=True,
        help="Path to onnx encoder"
    )
    parser.add_argument(
        "--decoder", 
        type=str, 
        required=True,
        help="Path to onnx decoder"
    )
    parser.add_argument(
        "--provider",
        choices=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    parser.add_argument(
        "--cmvn",
        type=str,
        required=True,
        help="Path to cmvn"
    )
    parser.add_argument(
        "--dict",
        type=str,
        required=True,
        help="Path to dict"
    )
    parser.add_argument(
        "--spm_model",
        type=str,
        required=True,
        help="Path to spm model"
    )
    parser.add_argument(
        "--wavlist",
        type=str,
        required=True,
        help="File to wav path list"
    )
    parser.add_argument(
        "--hypo",
        type=str,
        required=True,
        help="File of hypos"
    )
    
    return parser.parse_args()
    
    
def parse_wavlist(wavlist: str):
    wavpaths = []
    with open(wavlist) as f:
        for line in f:
            line = line.strip()
            if not os.path.exists(line):
                print(f"{line} doesn't exist.")
                continue
            wavpaths.append(line)
            
    return wavpaths
    

def main():
    args = parse_args()
    
    feat_extractor = ASRFeatExtractor(args.cmvn)
    tokenizer = ChineseCharEnglishSpmTokenizer(args.dict, args.spm_model)
    wavlist = parse_wavlist(args.wavlist)
    onnx_model = FireRedASROnnxModel(args.encoder, args.decoder, [args.provider])
    
    wf = open(args.hypo, "wt")
    
    total_duration = 0
    total_infer_time = 0
    for wav in wavlist:
        feats, lengths, durs = feat_extractor([wav])
        total_duration += durs[0]
        start_time = time.time()
        n_layer_cross_k, n_layer_cross_v, cross_attn_mask = onnx_model.run_encoder(
            to_numpy(feats),
            to_numpy(lengths)
        )
        
        hypo_ids = onnx_model.run_decoder(
            n_layer_cross_k,
            n_layer_cross_v,
            cross_attn_mask
        )
        hypo_text = tokenizer.detokenize(hypo_ids)
        end_time = time.time()
        batch_infer_time = end_time - start_time
        
        batch_ref = batch_infer_time / durs[0]
        total_infer_time += batch_infer_time
        
        print(f"{wav} -> {hypo_text}, ref: {batch_ref}")
        wf.write(f"{hypo_text} ({wav})\n")    
    
    rtf = total_infer_time / total_duration if total_duration > 0 else 0
    print(f"total_infer_time: {total_infer_time}")
    print(f"rtf: {rtf}")
    print("infer done.")
    

if __name__ == "__main__":
    main()