#!/bin/bash

model="pretrained_models/model.pth.tar"
encoder="onnx_encoder"
decoder="onnx_decoder"
encoder_int8="onnx_encoder_int8"
decoder_int8="onnx_decoder_int8"

python to_onnx.py \
--model $model \
--encoder $encoder \
--decoder $decoder \
--encoder_int8 $encoder_int8 \
--decoder_int8 $decoder_int8