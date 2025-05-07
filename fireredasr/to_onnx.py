import model_wrapper
from fireredasr.models.fireredasr import FireRedAsrAed

import torch
import onnx
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic
import numpy as np

import os
import argparse


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def load_model(model_path):
    package = torch.load(model_path, 
                         map_location=lambda storage, 
                         loc: storage, weights_only=False)
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model


def export_encoder(fireredasr_model, args):
    encoder = model_wrapper.AudioEncoderTensorCache(
        fireredasr_model.encoder, 
        fireredasr_model.decoder)
    encoder.eval()

    encoder_input = torch.randn(1, 418, 80)
    encoder_input_lengths = torch.tensor([418], dtype=torch.int64)
    n_layer_cross_k, n_layer_cross_v, cross_attn_mask = encoder(
        encoder_input, 
        encoder_input_lengths
    )

    if not os.path.exists(args.encoder):
        os.makedirs(args.encoder)
    onnx_encoder_file = os.path.join(args.encoder, "encoder.onnx")

    torch.onnx.export(
        encoder,
        (encoder_input, encoder_input_lengths),
        onnx_encoder_file,
        export_params=True,
        opset_version=args.opset_version,
        verbose=True,
        input_names=["encoder_input",
                     "encoder_input_lengths"],
        output_names=["n_layer_cross_k", 
                      "n_layer_cross_v",
                      "cross_attn_mask"],
        dynamic_axes={
            "encoder_input": {
                0: "batch_size",
                1: "input_length"
            },
            "encoder_input_lengths": {
                0: "batch_size"
            },
            "n_layer_cross_k": {
                1: "batch_size",
                2: "length"
            },
            "n_layer_cross_v": {
                1: "batch_size",
                2: "length"
            },
            "cross_attn_mask": {
                0: "batch_size",
                2: "length"
            }
        },
        external_data=True
    )

    onnx.checker.check_model(onnx_encoder_file)
    ort_session = onnxruntime.InferenceSession(onnx_encoder_file)
    onnx_encoder_input = to_numpy(encoder_input)
    onxx_encoder_input_lengths = to_numpy(encoder_input_lengths)
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_encoder_input,
                  ort_session.get_inputs()[1].name: onxx_encoder_input_lengths}
    ort_outputs = ort_session.run(None, ort_inputs)

    try:
        np.testing.assert_allclose(to_numpy(n_layer_cross_k), ort_outputs[0], rtol=1e-03, atol=1e-05)
    except AssertionError as e:
        print(e)
    try:
        np.testing.assert_allclose(to_numpy(n_layer_cross_v), ort_outputs[1], rtol=1e-03, atol=1e-05)
    except AssertionError as e:
        print(e)
    try:
        np.testing.assert_allclose(to_numpy(cross_attn_mask), ort_outputs[2], rtol=1e-03, atol=1e-05)
    except AssertionError as e:
        print(e)
        
    print("export onnx encoder done.")
    
    # Generate int8 quantization models
    # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection
    print("Generate int8 quantization models")
    
    if not os.path.exists(args.encoder_int8):
        os.mkdir(args.encoder_int8)
    onnx_encoder_int8_file = "encoder_int8.onnx"
    onnx_encoder_int8_file = os.path.join(args.encoder_int8, onnx_encoder_int8_file)
    quantize_dynamic(
        model_input=onnx_encoder_file,
        model_output=onnx_encoder_int8_file,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )
    
    return n_layer_cross_k, n_layer_cross_v, cross_attn_mask


def export_decoder(fireredasr_model, args,
                   n_layer_cross_k,
                   n_layer_cross_v,
                   cross_attn_mask):
    decoder = model_wrapper.TextDecoderTensorCache(
        fireredasr_model.decoder)
    decoder.eval()
    
    # 将 cross_attn_mask 转换为 decoder 中所需要的形式
    cross_attn_mask = cross_attn_mask.to(torch.float32)
    cross_attn_mask[cross_attn_mask == 0] = -torch.inf
    cross_attn_mask[cross_attn_mask == 1] = 0.0
    
    tokens = torch.tensor([[decoder.decoder.sos_id]] * 1, dtype=torch.int64)
    n_layer_self_k_cache = torch.zeros(
        (
            len(decoder.blocks),
            1,
            448,
            1280
        )
    )
    n_layer_self_v_cache = torch.zeros(
        (
            len(decoder.blocks),
            1,
            448,
            1280
        )
    )
    offset = torch.zeros(1, dtype=torch.int64)

    logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache, _, _, _, _ = decoder(
        tokens,
        n_layer_self_k_cache,
        n_layer_self_v_cache,
        n_layer_cross_k,
        n_layer_cross_v,
        cross_attn_mask,
        offset
    )

    if not os.path.exists(args.decoder):
        os.makedirs(args.decoder)
    onnx_decoder_file = os.path.join(args.decoder, "decoder.onnx")

    torch.onnx.export(
        decoder,
        (tokens,
        n_layer_self_k_cache,
        n_layer_self_v_cache,
        n_layer_cross_k,
        n_layer_cross_v,
        cross_attn_mask,
        offset),
        onnx_decoder_file,
        export_params=True,
        opset_version=args.opset_version,
        verbose=True,
        input_names=["tokens",
                     "in_n_layer_self_k_cache",
                     "in_n_layer_self_v_cache",
                     "in_n_layer_cross_k",
                     "in_n_layer_cross_v",
                     "in_cross_attn_mask",
                     "in_offset"],
        output_names=["logits",
                      "out_n_layer_self_k_cache",
                      "out_n_layer_self_v_cache",
                      "out_n_layer_cross_k",
                      "out_n_layer_cross_v",
                      "out_cross_attn_mask",
                      "out_offset"],
        dynamic_axes={
            "tokens": {0: "n_audio", 1: "n_tokens"},
            "in_n_layer_self_k_cache": {1: "n_audio"},
            "in_n_layer_self_v_cache": {1: "n_audio"},
            "in_n_layer_cross_k": {1: "n_audio", 2: "T"},
            "in_n_layer_cross_v": {1: "n_audio", 2: "T"},
            "in_cross_attn_mask": {0: "n_audio", 2: "T"},
        },
        external_data=True
    )

    onnx.checker.check_model(onnx_decoder_file)
    ort_session = onnxruntime.InferenceSession(onnx_decoder_file)

    onnx_tokens = to_numpy(tokens)
    onnx_n_layer_self_k_cache = to_numpy(n_layer_self_k_cache)
    onnx_n_layer_self_v_cache = to_numpy(n_layer_self_v_cache)
    onnx_n_layer_cross_k = to_numpy(n_layer_cross_k)
    onnx_n_layer_cross_v = to_numpy(n_layer_cross_v)
    onnx_offset = to_numpy(offset)
    onnx_cross_attn_mask = to_numpy(cross_attn_mask)

    ort_inputs = {ort_session.get_inputs()[0].name: onnx_tokens,
                  ort_session.get_inputs()[1].name: onnx_n_layer_self_k_cache,
                  ort_session.get_inputs()[2].name: onnx_n_layer_self_v_cache,
                  ort_session.get_inputs()[3].name: onnx_n_layer_cross_k,
                  ort_session.get_inputs()[4].name: onnx_n_layer_cross_v,
                  ort_session.get_inputs()[5].name: onnx_cross_attn_mask,
                  ort_session.get_inputs()[6].name: onnx_offset,}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    try:
        np.testing.assert_allclose(to_numpy(logits), ort_outputs[0], rtol=1e-03, atol=1e-05)
    except AssertionError as e:
        print(e)    
    try:
        np.testing.assert_allclose(to_numpy(out_n_layer_self_k_cache), ort_outputs[1], rtol=1e-03, atol=1e-05)
    except AssertionError as e:
        print(e)  
    try:
        np.testing.assert_allclose(to_numpy(out_n_layer_self_v_cache), ort_outputs[2], rtol=1e-03, atol=1e-05)
    except AssertionError as e:
        print(e)
    
    print("export onnx decoder done.")
    
    if not os.path.exists(args.decoder_int8):
        os.mkdir(args.decoder_int8)
    onnx_decoder_int8_file = "decoder_int8.onnx"
    onnx_decoder_int8_file = os.path.join(args.decoder_int8, onnx_decoder_int8_file)
    quantize_dynamic(
        model_input=onnx_decoder_file,
        model_output=onnx_decoder_int8_file,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="export FireRedASR-AED torch model to onnx")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to FireRedASR-AED torch model"
    )
    parser.add_argument(
        "--encoder", 
        type=str, 
        required=True,
        help="Dir to the exported onnx encoder"
    )
    parser.add_argument(
        "--decoder", 
        type=str, 
        required=True,
        help="Dir to the exported onnx decoder"
    )
    parser.add_argument(
        "--encoder_int8", 
        type=str, 
        required=True,
        help="Dir to the exported onnx encoder after int8 quantization"
    )
    parser.add_argument(
        "--decoder_int8", 
        type=str, 
        required=True,
        help="Dir to the exported onnx encoder after int8 quantization"
    )
    parser.add_argument(
        "--opset_version",
        type=int, 
        default=14,
        help="onnx opset version"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fireredasr_model = load_model(args.model)
    n_layer_cross_k, n_layer_cross_v, cross_attn_mask = export_encoder(fireredasr_model, args)
    export_decoder(fireredasr_model, args, n_layer_cross_k, n_layer_cross_v, cross_attn_mask)
    

if __name__ == "__main__":
    main()