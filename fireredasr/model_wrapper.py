import torch
import torch.nn as nn
from torch import Tensor

from fireredasr.models.module.conformer_encoder import ConformerEncoder
from fireredasr.models.module.transformer_decoder import (
    TransformerDecoder,
    DecoderLayer,
    DecoderMultiHeadAttention,
    DecoderScaledDotProductAttention,
    PositionalEncoding
)


def DecoderScaledDotProductAttentionForward(
    self: DecoderScaledDotProductAttention,
    q: Tensor, 
    k: Tensor,
    v: Tensor,
    mask: Tensor
):
    attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
    if mask is not None:
        # mask is such as [[[0, 0, 0, 0, ..., -inf, -inf]]]
        attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
    else:
        attn = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)
    return output

def PositionalEncodingForward(self, offset: Tensor):
    length = offset[0]
    return self.pe[:, :length].clone().detach()


DecoderScaledDotProductAttention.forward = DecoderScaledDotProductAttentionForward
PositionalEncoding.forward = PositionalEncodingForward


class AudioEncoderTensorCache(nn.Module):
    def __init__(self, 
                 encoder: ConformerEncoder, 
                 decoder: TransformerDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input: Tensor, input_length: Tensor):
        encoder_output, _, encoder_mask = self.encoder(input, input_length)
        
        n_layer_cross_k_list = []
        n_layer_cross_v_list = []
        
        for layer in self.decoder.layer_stack:
            # layer: DecoderLayer
            n_layer_cross_k_list.append(layer.cross_attn.w_ks(encoder_output))
            n_layer_cross_v_list.append(layer.cross_attn.w_vs(encoder_output))
        
        # 转换为 decoder 的形式
        encoder_mask = encoder_mask.to(torch.float32)
        encoder_mask[encoder_mask == 0] = -torch.inf
        encoder_mask[encoder_mask == 1] = 0.0

        return (torch.stack(n_layer_cross_k_list),
                torch.stack(n_layer_cross_v_list),
                encoder_mask)


class DecoderMultiHeadSelfAttention(nn.Module):
    def __init__(self, multiHeadSelfAttention: DecoderMultiHeadAttention):
        super().__init__()
        self.multiHeadSelfAttention = multiHeadSelfAttention
        
    def forward(self, 
                x: Tensor,
                k_cache: Tensor,
                v_cache: Tensor,
                mask: Tensor):
        bs = x.size(0)

        # 当前时间步为 t
        # k_cache 和 v_cache 是 时间步 [0: t-1] 的 self_attn_k 和 self_attn_v 的缓存
        q = self.multiHeadSelfAttention.w_qs(x)
        k = self.multiHeadSelfAttention.w_ks(x)
        v = self.multiHeadSelfAttention.w_vs(x)

        k_cache[:, -k.shape[1] :, :] = k
        v_cache[:, -v.shape[1] :, :] = v

        q = q.view(bs, -1, self.multiHeadSelfAttention.n_head, self.multiHeadSelfAttention.d_k)
        k = k_cache.view(bs, -1, self.multiHeadSelfAttention.n_head, self.multiHeadSelfAttention.d_k)
        v = v_cache.view(bs, -1, self.multiHeadSelfAttention.n_head, self.multiHeadSelfAttention.d_k)
        k = k.view(bs, -1, self.multiHeadSelfAttention.n_head, self.multiHeadSelfAttention.d_k)
        v = v.view(bs, -1, self.multiHeadSelfAttention.n_head, self.multiHeadSelfAttention.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output = self.multiHeadSelfAttention.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.multiHeadSelfAttention.d_model)
        output = self.multiHeadSelfAttention.fc(output)
        output = self.multiHeadSelfAttention.dropout(output)

        return output, k_cache, v_cache
    

class DecoderMultiHeadCrossAttention(nn.Module):
    def __init__(self, multiHeadCrossAttention: DecoderMultiHeadAttention):
        super().__init__()
        self.multiHeadCrossAttention = multiHeadCrossAttention
        
    def forward(self,
                x: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Tensor):
        # k 和 v 为缓存，不需要每次都进行 self_k 和 self_v 的投影计算
        bs = x.size(0)
        x = self.multiHeadCrossAttention.w_qs(x)
        x = x.view(bs, -1, self.multiHeadCrossAttention.n_head, self.multiHeadCrossAttention.d_k)
        k = k.view(bs, -1, self.multiHeadCrossAttention.n_head, self.multiHeadCrossAttention.d_k)
        v = v.view(bs, -1, self.multiHeadCrossAttention.n_head, self.multiHeadCrossAttention.d_k)

        x = x.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)

        output = self.multiHeadCrossAttention.attention(x, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.multiHeadCrossAttention.d_model)
        output = self.multiHeadCrossAttention.fc(output)
        output = self.multiHeadCrossAttention.dropout(output)

        return output


class ResidualAttentionBlockTensorCache(nn.Module):
    def __init__(self, decoder_layer: DecoderLayer):
        super().__init__()
        self.original_decoder_layer = decoder_layer
        self.self_attn = DecoderMultiHeadSelfAttention(decoder_layer.self_attn)
        self.cross_attn = DecoderMultiHeadCrossAttention(decoder_layer.cross_attn)
        
    def forward(self,
                x: Tensor,
                self_k_cache: Tensor,
                self_v_cache: Tensor,
                cross_k: Tensor,
                cross_v: Tensor,
                self_attn_mask: Tensor,
                cross_attn_mask: Tensor):
        # q.shape (B, 1, dim)
        x_self_attn_norm = self.original_decoder_layer.self_attn_norm(x)
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.self_attn(
            x_self_attn_norm, self_k_cache, self_v_cache, self_attn_mask)
        
        # residual
        x = x + self_attn_x
        
        residual = x
        x_cross_attn_norm = self.original_decoder_layer.cross_attn_norm(x)
        x_cross_attn = self.cross_attn(x_cross_attn_norm, cross_k, cross_v, cross_attn_mask)
        x = residual + x_cross_attn

        x = x + self.original_decoder_layer.mlp(self.original_decoder_layer.mlp_norm(x))
        
        return x, self_k_cache_updated, self_v_cache_updated
        

class TextDecoderTensorCache(nn.Module):
    def __init__(self, decoder: TransformerDecoder):
        super().__init__()
        self.decoder = decoder
        
        self.blocks = []
        for original_layer in self.decoder.layer_stack:
            self.blocks.append(
                ResidualAttentionBlockTensorCache(original_layer))
        
    def forward(self, 
                tokens: Tensor,
                n_layer_self_k_cache: Tensor,
                n_layer_self_v_cache: Tensor,
                n_layer_cross_k: Tensor,
                n_layer_cross_v: Tensor,
                cross_attn_mask: Tensor,
                offset: Tensor):
        # n_layer_self_k_cache.shape (n_layer, batch_size, seq_len, dim)
        # n_layer_cross_k.shape (n_layer, batch_size, encoder_output_len, dim)
        # offset.shape (1)
        # offset 表示已经预测了多少个token
        
        # FIXME 目前只支持 batch size 为 1 的情况
        # self_attn_mask = torch.empty(
        #     1, tokens.shape[-1], tokens.shape[-1]).fill_(-torch.inf).triu_(1)
        self_attn_mask = torch.empty(
            1, offset[0] + 1, offset[0] + 1).fill_(-torch.inf).triu_(1)
        self_attn_mask = self_attn_mask[:, -1:, :]

        # x = self.decoder.dropout(
        #     self.decoder.tgt_word_emb(tokens) * self.decoder.scale +
        #     self.decoder.positional_encoding(tokens)
        # )
        x = self.decoder.dropout(
            self.decoder.tgt_word_emb(tokens) * self.decoder.scale +
            self.decoder.positional_encoding(offset + 1)
        )
        x = x[:, -1:, :]
        tokens = tokens[:, -1:]
        idx = tokens[0][0]
        
        i = 0
        for block in self.blocks:
            self_k_cache = n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :]
            self_v_cache = n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :]
            # self_k_cache = n_layer_self_k_cache[i, :, : offset[0] + 1, :]
            # self_v_cache = n_layer_self_v_cache[i, :, : offset[0] + 1, :]
            x, self_k_cache, self_v_cache = block(
                x,
                self_k_cache,
                self_v_cache,
                n_layer_cross_k[i],
                n_layer_cross_v[i],
                self_attn_mask,
                cross_attn_mask
            )
            n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_k_cache
            n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_v_cache
            # n_layer_self_k_cache[i, :, : offset[0] + 1, :] = self_k_cache
            # n_layer_self_v_cache[i, :, : offset[0] + 1, :] = self_v_cache
            i += 1

        output = self.decoder.layer_norm_out(x)
        logits = self.decoder.tgt_word_prj(output)

        return (logits, n_layer_self_k_cache, n_layer_self_v_cache,
                n_layer_cross_k, n_layer_cross_v, cross_attn_mask, offset)