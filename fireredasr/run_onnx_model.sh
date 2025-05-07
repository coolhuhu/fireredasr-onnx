#!/bin/bash

encoder_path="onnx_encoder_int8/encoder_int8.onnx"
decoder_path="onnx_decoder_int8/decoder_int8.onnx"
cmvn_path="pretrained_models/cmvn.ark"
dict_path="pretrained_models/dict.txt"
spm_model="pretrained_models/train_bpe1000.model"
wavlist="wavlist.txt"
hypo="hypos.txt"

python test_onnx_model.py \
--encoder $encoder_path \
--decoder $decoder_path \
--cmvn $cmvn_path \
--dict $dict_path \
--spm_model $spm_model \
--wavlist $wavlist \
--hypo $hypo \
--provider "CPUExecutionProvider"