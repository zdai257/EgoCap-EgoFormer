# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter
from .ViT_encoder import ViTEncoder


class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        self.embeddings = DecoderEmbeddings(config)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, tgt, tgt_mask):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        tgt = self.embeddings(tgt).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask,
                          pos=pos_embed, query_pos=query_embed,
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))

        return hs


class EgoTransformer(Transformer):
    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__(config, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout,
                 activation, normalize_before, return_intermediate_dec)
        # Define learnable embedding for "tag_token"
        self.context_embeddings = ContextEmbeddings(config)
        # Redefine decoder layer with ContextFuse
        decoder_layer = EgoTransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = EgoTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                             return_intermediate=return_intermediate_dec)
        self._reset_parameters()

    def forward(self, src, mask, pos_embed, tgt, tgt_mask, tag_token, tag_mask):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # 'pos_embed' is only for Encoder
        mask = mask.flatten(1)

        tgt = self.embeddings(tgt).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)
        # tag_token: (B, seq") => (seq", B, d_m); tag_mask: (B, seq") as <key_padding_mask>
        tag_token = self.context_embeddings(tag_token).permute(1, 0, 2)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # memory: (seq, B, d_m); mask: (B, seq);
        # tgt: (seq', B, d_m); squared_tgt_mask: (seq', seq')
        hs = self.decoder(tgt, memory, tag_token, tag_mask=tag_mask,
                          memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask,
                          pos=pos_embed, query_pos=query_embed,
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
        return hs


# Dual-ViT model
class EgoFormer(Transformer):
    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__(config, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout,
                 activation, normalize_before, return_intermediate_dec)
        # Define learnable embedding for "tag_token"
        #self.context_embeddings = ContextEmbeddings(config)
        # Extract Context-ViT
        vit_encoder2 = ViTEncoder()
        #self.ctx_encoder = IntermediateLayerGetter(vit_encoder2, return_layers={"classifier0": "ctx2"})
        self.ctx_encoder1 = IntermediateLayerGetter(vit_encoder2, return_layers={"body": "body2"})
        self.ctx_encoder2 = nn.Linear(in_features=768, out_features=d_model)

        # Redefine decoder layer with ContextFuse
        decoder_layer = EgoViTDecoderLayer(d_model, nhead, dim_feedforward,
                                           dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = EgoViTDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                     return_intermediate=return_intermediate_dec)
        self._reset_parameters()

    def forward(self, src, mask, pos_embed, tgt, tgt_mask, img):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # 'pos_embed' is only for Encoder
        mask = mask.flatten(1)

        tgt = self.embeddings(tgt).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)
        # tag_token: (B, seq") => (seq", B, d_m); tag_mask: (B, seq") as <key_padding_mask>
        #tag_token = self.context_embeddings(tag_token).permute(1, 0, 2)
        # input['pixel_values'] -> ctx

        xs = self.ctx_encoder1(img)
        xs_key = xs[next(iter(xs))]
        xs = xs_key[next(iter(xs_key))].last_hidden_state
        ctx = F.relu(self.ctx_encoder2(xs))

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # memory: (seq, B, d_m); mask: (B, seq);
        # tgt: (seq', B, d_m); squared_tgt_mask: (seq', seq')
        # ctx: (seq", B, d_m) ?
        hs = self.decoder(tgt, memory, ctx.permute(1, 0, 2), ctx_mask=None,
                          memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask,
                          pos=pos_embed, query_pos=query_embed,
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
        return hs


# SEPARATE ENCODER & DECODER
class Encoder(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten BxCxHxW or BxCxHxWxT (video) to SeqxBxC

        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        #print("Input X size = ", src.shape)
        #print(mask.shape, pos_embed.shape)
        # Utilize 'src_key_padding_mask' as Temporal mask?
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        #print("Encoder out shape = ", memory.shape)
        return memory, mask, pos_embed


# SEPARATE ENCODER & DECODER
class Decoder(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        self.embeddings = DecoderEmbeddings(config)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, tgt, tgt_mask, mask=None, pos_embed=None):

        seq, bs, d_model = memory.shape

        #print(tgt.shape, tgt_mask.shape)
        tgt = self.embeddings(tgt).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        #print(query_embed.shape)
        query_embed = query_embed.repeat(1, bs, 1)

        #print(tgt.shape, memory.shape)
        #print(query_embed.shape, generate_square_subsequent_mask(len(tgt)).shape)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask,
                          pos=pos_embed, query_pos=query_embed,
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))

        #print("out shape = ", hs.shape)
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class EgoTransformerDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__(decoder_layer=decoder_layer, num_layers=num_layers, norm=norm,
                         return_intermediate=return_intermediate)

    def forward(self, tgt, memory, tag_token, tag_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tag_token, tag_mask=tag_mask,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class EgoViTDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__(decoder_layer=decoder_layer, num_layers=num_layers, norm=norm,
                         return_intermediate=return_intermediate)

    def forward(self, tgt, memory, ctx, ctx_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, ctx, ctx_mask=ctx_mask,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class EgoTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                 activation, normalize_before)
        # Implementation of Stacked MHA layer for ContextFuse
        self.multihead_attn2 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.norm20 = nn.LayerNorm(d_model)
        self.dropout20 = nn.Dropout(dropout)

    def forward_post(self, tgt, memory, tag_token, tag_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # Cross MHA with tag_token
        tgt2 = self.multihead_attn2(query=tgt, key=tag_token, value=tag_token,
                                    attn_mask=None, key_padding_mask=tag_mask)[0]
        tgt = tgt + self.dropout20(tgt2)
        tgt = self.norm20(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tag_token, tag_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        # Cross MHA with tag_token
        tgt2 = self.norm20(tgt)
        tgt2 = self.multihead_attn2(query=tgt2, key=tag_token, value=tag_token,
                                    attn_mask=None, key_padding_mask=tag_mask)[0]
        tgt = tgt + self.dropout20(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tag_token, tag_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tag_token, tag_mask, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tag_token, tag_mask, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class EgoViTDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                 activation, normalize_before)
        # Implementation of Stacked MHA layer for ContextViT fusion
        self.multihead_attn2 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.norm20 = nn.LayerNorm(d_model)
        self.dropout20 = nn.Dropout(dropout)

    def forward_post(self, tgt, memory, ctx, ctx_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # Cross MHA with ctx
        tgt2 = self.multihead_attn2(query=tgt, key=ctx, value=ctx,
                                    attn_mask=None, key_padding_mask=ctx_mask)[0]
        tgt = tgt + self.dropout20(tgt2)
        tgt = self.norm20(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, ctx, ctx_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        # Cross MHA with tag_token
        tgt2 = self.norm20(tgt)
        tgt2 = self.multihead_attn2(query=tgt2, key=ctx, value=ctx,
                                    attn_mask=None, key_padding_mask=ctx_mask)[0]
        tgt = tgt + self.dropout20(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, ctx, ctx_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, ctx, ctx_mask, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, ctx, ctx_mask, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class DecoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ContextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        embeddings = self.context_embeddings(x)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def build_transformer(config):
    if config.modality == 'ego':
        return EgoFormer(
            config,
            d_model=config.hidden_dim,
            dropout=config.dropout,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            num_encoder_layers=config.enc_layers,
            num_decoder_layers=config.dec_layers,
            normalize_before=config.pre_norm,
            return_intermediate_dec=False,
        )
    else:
        return Transformer(
            config,
            d_model=config.hidden_dim,
            dropout=config.dropout,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            num_encoder_layers=config.enc_layers,
            num_decoder_layers=config.dec_layers,
            normalize_before=config.pre_norm,
            return_intermediate_dec=False,
        )
