import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer, Encoder, Decoder


class Caption(nn.Module):
    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(
            backbone.num_channels, hidden_dim, kernel_size=1)
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

    def forward(self, samples, target, target_mask):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask,
                              pos[-1], target, target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class EgoCaption(Caption):
    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__(backbone, transformer, hidden_dim, vocab_size)

    def forward(self, samples, target, target_mask, tag_token, tag_mask):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask,
                              pos[-1], target, target_mask, tag_token, tag_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class EgoViT(Caption):
    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__(backbone, transformer, hidden_dim, vocab_size)

    def forward(self, samples, target, target_mask, img):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask,
                              pos[-1], target, target_mask, img)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


# New Model
class CaptionWithEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, max_position_embeddings, start_token=101, end_token=102, vocab_size=30522):
        super(CaptionWithEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_position_embeddings = max_position_embeddings
        self.start_token = start_token
        self.end_token = end_token
        self.vocab_size = vocab_size
        self.diverse_mask = None

    def forward(self, samples, target, target_mask, mask=None, pos_embed=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        mem, enc_mask, pos_embed = self.encoder(samples)
        out = self.decoder(mem, target, target_mask, mask=enc_mask, pos_embed=pos_embed)

        return out

    def decode(self, samples, target, target_mask, mask=None, pos_embed=None, beam_width=None, diverse_m=None):
        mem, enc_mask, pos_embed = self.encoder(samples)

        if beam_width is not None:

            output, outputs = [], []
            if diverse_m is not None and isinstance(diverse_m, int) and diverse_m > 1:
                self.diverse_mask = torch.ones((samples.shape[0], self.max_position_embeddings, self.vocab_size))
                for m in range(diverse_m):
                    decoded_batch, decoded_batch_beams = self.beam_decode(mem, target, target_mask, mask=enc_mask,
                                                                          pos_embed=pos_embed, beam_width=beam_width,
                                                                          diverse_mask=self.diverse_mask)
                    output.append(decoded_batch)
                    outputs.append(decoded_batch_beams)

                    decoded_mask = decoded_batch_beams.numpy()
                    # Prevent same token appear in same (& neighbor) position as previous rounds of Beam Searches
                    for index, val in np.ndenumerate(decoded_mask):
                        if val == 0 or val == self.start_token or val == self.end_token:
                            continue
                        for mask_seq_index in range(index[2] - 2, index[2] + 2):
                            if mask_seq_index >= 0 and mask_seq_index < self.max_position_embeddings - 1:
                                self.diverse_mask[index[0], mask_seq_index, int(decoded_mask[index[0], index[1], index[2]])] = 0

                return output[0], outputs
            else:
                output, outputs = self.beam_decode(mem, target, target_mask, mask=enc_mask, pos_embed=pos_embed,
                                                   beam_width=beam_width)
                return output, outputs.unsqueeze(0)
        else:
            return self.greedy_decode(mem, target, target_mask, mask=enc_mask, pos_embed=pos_embed), None

    def greedy_decode(self, encoder_outputs, caption, cap_mask, mask=None, pos_embed=None):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        batch_size, seq_len = caption.shape
        decoded_batch = torch.zeros((batch_size, seq_len))
        #decoded_batch[:, 0] = 101

        # [<CLS>, <PAD>, <PAD> ...]
        caption = torch.zeros((batch_size, self.max_position_embeddings), dtype=torch.long)
        cap_mask = torch.ones((batch_size, self.max_position_embeddings), dtype=torch.bool)

        caption[:, 0] = self.start_token
        # tgt_mask refreshed at iter!
        cap_mask[:, 0] = False

        #print(caption.shape)

        for t in range(seq_len):
            decoder_output = self.decoder(encoder_outputs, caption, cap_mask, mask=mask, pos_embed=pos_embed)

            predictions = decoder_output[:, t, :]
            predicted_id = torch.argmax(predictions, dim=-1)  # topk = 1

            if predicted_id[0] == self.end_token or t >= seq_len-1:
                decoded_batch[:, t] = self.end_token
                break

            caption[:, t + 1] = predicted_id[0]
            cap_mask[:, t + 1] = False

            decoded_batch[:, t] = predicted_id[0]

        return decoded_batch

    #@timeit
    def beam_decode(self, encoder_outputs, caption, cap_mask, mask=None, pos_embed=None, beam_width=3,
                    mode="mul_score", diverse_mask=None):

        batch_size, seq_len = caption.shape
        decoded_batch = torch.zeros((batch_size, seq_len))
        decoded_batch_beams = torch.zeros((batch_size, beam_width, seq_len))

        # [<CLS>, <PAD>, <PAD> ...]
        caption = torch.zeros((batch_size, self.max_position_embeddings), dtype=torch.long)
        cap_mask = torch.ones((batch_size, self.max_position_embeddings), dtype=torch.bool)
        caption[:, 0] = self.start_token
        # tgt_mask refreshed at iter!
        cap_mask[:, 0] = False

        for index in range(batch_size):
            k = beam_width

            # Tensor to store top k sequences' scores; now they're just <sos> & 0s
            k_prev_words = caption[index].unsqueeze(0).expand(k, caption.shape[-1])
            # seqs: (k, 1)
            seqs = k_prev_words.detach().clone()[:, 0:1]
            # k_cap_mask: (k, seq_len)
            k_cap_mask = cap_mask[index].unsqueeze(0).expand(k, cap_mask.shape[-1])
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # decoding goes sentence by sentence
            for t in range(seq_len - 1):

                # k_encoder_output: (enc_seq_len, k, d_model)
                k_encoder_output = encoder_outputs[:, index:index + 1, :].expand(encoder_outputs.shape[0], k,
                                                                                 encoder_outputs.shape[2])

                # decoder_output: (k, seq_len, vocab_size); k_prev_words: (k, seq_len); k_cap_mask: (k, seq_len)
                decoder_output = self.decoder(k_encoder_output, k_prev_words, k_cap_mask,
                                      mask=mask[index].unsqueeze(0).expand(k, mask.shape[-1]),
                                      pos_embed=pos_embed[:, index:index+1, :].expand(pos_embed.shape[0], k, pos_embed.shape[2]))
                vocab_size = decoder_output.shape[-1]

                scores = decoder_output[:, t, :]  # (k, vocab_size)

                # Accumulate scores or Multiply?
                if mode == "mul_score":
                    scores = F.log_softmax(scores, dim=1)
                    if diverse_mask is not None:
                        # make 0 scores at (index, t, vocab_size==0)
                        scores = diverse_mask[index, t].expand_as(scores) * scores
                        #print(torch.where(scores == 0))
                    scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)
                else:
                    scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)

                # For the first step, all k points will have the same scores
                if t == 0:
                    # in: (k, vocal_size)
                    top_k_scores, top_k_indices = scores[0].topk(k, dim=-1)  # (k,)
                else:
                    # in: (k, vocal_size)
                    # KEY: Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_indices = scores.view(-1).topk(k, dim=-1)  # (k,)

                #print(top_k_scores)
                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_indices / vocab_size  # (k)
                next_word_inds = top_k_indices % vocab_size  # (k)

                # Add new words to sequences: cat based on seqs of (k, t+1) => (k, t+2)
                seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (k, t + 2)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != self.end_token]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                #print(incomplete_inds, complete_inds)
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break

                seqs = seqs[incomplete_inds]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                if t < seq_len - 2:
                    k_prev_words = k_prev_words[incomplete_inds]
                    k_cap_mask = k_cap_mask[incomplete_inds]

                    k_prev_words[:, t + 1] = next_word_inds[incomplete_inds]
                    k_cap_mask[:, t + 1] = False
                else:
                    k_prev_words = k_prev_words[incomplete_inds]
                    k_prev_words[:, t] = self.end_token

                    complete_seqs.extend(seqs[incomplete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[incomplete_inds])
                    break

            # Get the seq which Maximize cumulative score
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            # Get top k sequences
            top_ki = sorted(range(len(complete_seqs_scores)), key=lambda x: complete_seqs_scores[x], reverse=True)[-beam_width:]
            sequence = []
            for i in top_ki:
                sequence.append(complete_seqs[i])

            decoded_batch[index, :len(seq)] = torch.Tensor(seq)

            for beam in range(beam_width):
                decoded_batch_beams[index, beam, :len(sequence[beam])] = torch.Tensor(sequence[beam])

        return decoded_batch, decoded_batch_beams


class CaptionWithVideoEncoder(nn.Module):
    def __init__(self, backbone, Encoder, hidden_dim):
        super(CaptionWithVideoEncoder, self).__init__()
        self.backbone = backbone
        # Channel dimension reduction!
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.Encoder = Encoder

    def forward(self, samples):
        if not isinstance(samples, NestedTensor):
            #samples = nested_tensor_from_tensor_list(samples)
            raise TypeError("Haven't converted input <samples> to NestedTensor!")

        T = int(samples.tensors.shape[-1])
        # Samples (video input) size scaled to <= 299 per side, of size (B, 3, <=299, <=299, T)
        src_lst, mask_lst, pos_lst = [], [], []
        for sample_idx in range(T):
            sample, sample_m = samples.decompose()
            sample = NestedTensor(sample[:, :, :, :, sample_idx], sample_m[:, :, :, sample_idx])
            # Get feature_embedding & pos_encoding of each frame
            feature0, pos0 = self.backbone(sample)
            # Length of List of tensor/NestedTensor always 1
            src0, mask0 = feature0[-1].decompose()
            src_lst.append(self.input_proj(src0).unsqueeze(-1))
            mask_lst.append(mask0.unsqueeze(-1))
            pos_lst.append(pos0[-1].unsqueeze(-1))

        src = torch.cat(src_lst, dim=-1)
        mask = torch.cat(mask_lst, dim=-1)
        pos = torch.cat(pos_lst, dim=-1)
        assert mask is not None

        #print(pos.max(), pos.min())  # pos values in range [-1, 1]
        pos_max, pos_min = 1, -1
        pos_encode = torch.empty(pos.shape, dtype=pos.dtype, device=samples.tensors.device)
        for t in range(T):
            pos_encode[:, :, :, :, t] = pos[:, :, :, :, t] / T + pos_min + (t+0.5) * (pos_max - pos_min) / T

        # Input-tensor: {X (B, 256, 19, 19, 5), mask_all_False (B, 256, 19, 19, 5), pos_encode (B, 256, 19, 19, 5)}
        # caption (B, 128), cap_mask (B, 128)

        memory, enc_mask, pos_embed = self.Encoder(src, mask, pos_encode)

        return memory, enc_mask, pos_embed


class CaptionWithEncoder(nn.Module):
    def __init__(self, backbone, Encoder, hidden_dim):
        super(CaptionWithEncoder, self).__init__()
        self.backbone = backbone
        # Channel dimension reduction!
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.Encoder = Encoder

    def forward(self, samples):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # samples (img input) size scaled to <= 299 per side, of size (B, 3, <=299, <=299)
        features, pos = self.backbone(samples)
        #print(len(features), features[0].tensors.shape)
        #print(len(pos), pos[0].shape)
        # length of List of tensor/NestedTensor always 1
        src, mask = features[-1].decompose()
        #print(src.shape, mask.shape)

        assert mask is not None

        # Input-tensor: {X (B, 256, 14, 19), mask_all_False (B, 256, 14, 19), pos_encode (B, 256, 14, 19),
        # caption (B, 128), cap_mask (B, 128)
        memory, enc_mask, pos_embed = self.Encoder(self.input_proj(src), mask, pos[-1])

        return memory, enc_mask, pos_embed


class CaptionWithDecoder(nn.Module):
    def __init__(self, Decoder, hidden_dim, vocab_size):
        super().__init__()

        self.Decoder = Decoder
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=512, output_dim=vocab_size, num_layers=3)

    def forward(self, memory, tgt, tgt_mask, mask=None, pos_embed=None):
        seq, bs, d_model = memory.shape

        hs = self.Decoder(memory, tgt, tgt_mask=tgt_mask, mask=mask, pos_embed=pos_embed)

        out = self.mlp(hs.permute(1, 0, 2))
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config):
    backbone = build_backbone(config)
    transformer = build_transformer(config)

    model = Caption(backbone, transformer, config.hidden_dim, config.vocab_size)
    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion


def build_model_ego(config):
    backbone = build_backbone(config)
    transformer = build_transformer(config)

    model = EgoCaption(backbone, transformer, config.hidden_dim, config.vocab_size)
    criterion = torch.nn.CrossEntropyLoss()
    return model, criterion


def build_model_egovit(config):
    backbone = build_backbone(config)
    transformer = build_transformer(config)

    model = EgoViT(backbone, transformer, config.hidden_dim, config.vocab_size)
    criterion = torch.nn.CrossEntropyLoss()
    return model, criterion


def build_model_bs(config):
    backbone = build_backbone(config)

    my_encoder = Encoder(config, d_model=config.hidden_dim, nhead=config.nheads, num_encoder_layers=config.enc_layers,
                         dim_feedforward=config.dim_feedforward, dropout=config.dropout)
    my_decoder = Decoder(config, d_model=config.hidden_dim, nhead=config.nheads, num_decoder_layers=config.dec_layers,
                         dim_feedforward=config.dim_feedforward, dropout=config.dropout,
                         activation="relu", normalize_before=config.pre_norm, return_intermediate_dec=False)
    if config.modality == 'image':
        cap_encoder = CaptionWithEncoder(backbone, my_encoder, config.hidden_dim)
    else:
        cap_encoder = CaptionWithVideoEncoder(backbone, my_encoder, config.hidden_dim)
    cap_decoder = CaptionWithDecoder(my_decoder, config.hidden_dim, config.vocab_size)

    model = CaptionWithEncoderDecoder(cap_encoder, cap_decoder, config.max_position_embeddings, vocab_size=config.vocab_size)

    criterion = torch.nn.CrossEntropyLoss()
    #print(model)

    return model, criterion
