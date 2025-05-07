import model_wrapper
from fireredasr.models.fireredasr import FireRedAsrAed
from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer

import torch
import numpy as np

import os


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def load_model(model_path):
    package = torch.load(model_path, 
                         map_location=lambda storage, 
                         loc: storage, weights_only=False)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model


def main():
    decoder_max_len = 448
    model_dir = "/Users/lianghu/code/FireRedASRONNX/fireredasr/pretrained_models"

    cmvn_path = os.path.join(model_dir, "cmvn.ark")
    feat_extractor = ASRFeatExtractor(cmvn_path)

    batch_uttid = ["TEST_MEETING_T0000000001_S00000"]
    batch_wav_path = ["/Users/lianghu/code/FireRedASRONNX/fireredasr/wav/TEST_MEETING_T0000000001_S00000.wav"]
    feats, lengths, durs = feat_extractor(batch_wav_path)
    
    model_path = "/Users/lianghu/code/FireRedASRONNX/fireredasr/pretrained_models/model.pth.tar"
    fireredasr_model = load_model(model_path)
    encoder = model_wrapper.AudioEncoderTensorCache(
        fireredasr_model.encoder, 
        fireredasr_model.decoder)
    decoder = model_wrapper.TextDecoderTensorCache(
        fireredasr_model.decoder)
    encoder.eval()
    decoder.eval()

    n_layer_cross_k, n_layer_cross_v, cross_attn_mask = encoder(feats, lengths)

    prediction_tokens = torch.tensor([[decoder.decoder.sos_id]] * 1, dtype=torch.int64)
    offset = torch.zeros(1, dtype=torch.int64)
    n_layer_self_k_cache = torch.zeros(
        len(decoder.blocks),
        len(batch_wav_path),
        decoder_max_len,
        1280
    )
    n_layer_self_v_cache = torch.zeros(
        len(decoder.blocks),
        len(batch_wav_path),
        decoder_max_len,
        1280
    )

    results = [decoder.decoder.sos_id]
    token = torch.tensor([[decoder.decoder.sos_id]], dtype=torch.int64)
    for i in range(decoder_max_len):
        logits, n_layer_self_k_cache, n_layer_self_v_cache, _, _, _, _= decoder(
            token,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            cross_attn_mask,
            offset
        )
        offset += 1
        logits = logits.squeeze(0).argmax(dim=-1)
        p_token = logits[0].item()
        results.append(p_token)
        prediction_tokens = torch.tensor([results], dtype=torch.int64)
        token = torch.tensor([[p_token]], dtype=torch.int64)
        
        if p_token == decoder.decoder.eos_id:
            break

    print(results)

    dict_path =os.path.join(model_dir, "dict.txt")
    spm_model = os.path.join(model_dir, "train_bpe1000.model")
    tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
    text = tokenizer.detokenize(results)
    print(text)


if __name__ == "__main__":
    main()