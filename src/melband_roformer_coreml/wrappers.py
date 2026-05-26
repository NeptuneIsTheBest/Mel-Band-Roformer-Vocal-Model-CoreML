from __future__ import annotations

import torch
import torch.nn.functional as F
from ml_collections import ConfigDict
from torch import nn


class FixedRotaryEmbedding(nn.Module):
    def __init__(
        self,
        base_freqs: torch.Tensor,
        *,
        outer_batch: int,
        heads: int,
        seq_len: int,
        dim: int,
    ) -> None:
        super().__init__()
        positions = torch.arange(seq_len, dtype=base_freqs.dtype)
        freqs = torch.einsum("n,f->nf", positions, base_freqs.detach().cpu())
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos", freqs.cos().reshape(1, 1, seq_len, dim), persistent=False)
        self.register_buffer("sin", freqs.sin().reshape(1, 1, seq_len, dim), persistent=False)
        self.outer_batch = outer_batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.half_dim = dim // 2

    def rotate_queries_or_keys(self, t: torch.Tensor) -> torch.Tensor:
        x = t.reshape(self.outer_batch, self.heads, self.seq_len, self.half_dim, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        rotated = torch.stack((-x2, x1), dim=-1)
        rotated = rotated.reshape(self.outer_batch, self.heads, self.seq_len, self.dim)
        return (t * self.cos) + (rotated * self.sin)


class FixedAttention(nn.Module):
    def __init__(self, attention: nn.Module, *, outer_batch: int, seq_len: int) -> None:
        super().__init__()
        self.heads = attention.heads
        self.dim_head = attention.to_qkv.out_features // (3 * attention.heads)
        self.inner_dim = self.heads * self.dim_head
        self.outer_batch = outer_batch
        self.seq_len = seq_len
        self.scale = self.dim_head ** -0.5

        self.norm = attention.norm
        self.to_qkv = attention.to_qkv
        self.to_gates = attention.to_gates
        self.to_out = attention.to_out
        self.rotary_embed = attention.rotary_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(self.outer_batch, self.seq_len, 3, self.heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        gates = self.to_gates(x).sigmoid()
        gates = gates.permute(0, 2, 1).reshape(self.outer_batch, self.heads, self.seq_len, 1)
        out = out * gates

        out = out.permute(0, 2, 1, 3).reshape(self.outer_batch, self.seq_len, self.inner_dim)
        return self.to_out(out)


class RoformerCoreWrapper(nn.Module):
    def __init__(self, model: nn.Module, frames: int = 801) -> None:
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimators = model.mask_estimators
        self.frames = frames
        self.bands = len(model.band_split.dim_inputs)
        self.dim = self.layers[0][0].layers[0][0].to_gates.in_features

    def forward(self, packed_stft: torch.Tensor) -> torch.Tensor:
        x = self.band_split(packed_stft)

        for time_transformer, freq_transformer in self.layers:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.reshape(self.bands, self.frames, self.dim)
            x = time_transformer(x)

            x = x.reshape(1, self.bands, self.frames, self.dim)
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.reshape(self.frames, self.bands, self.dim)
            x = freq_transformer(x)

            x = x.reshape(1, self.frames, self.bands, self.dim)

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        return masks[:, 0]


class FixedWaveformToWaveformWrapper(nn.Module):
    def __init__(self, model: nn.Module, config: ConfigDict, frames: int) -> None:
        super().__init__()
        self.roformer_core = RoformerCoreWrapper(model, frames=frames)

        self.audio_channels = int(model.audio_channels)
        self.chunk_size = int(config.inference.chunk_size)
        self.n_fft = int(config.model.stft_n_fft)
        self.hop_length = int(config.model.stft_hop_length)
        self.frames = int(frames)
        self.freqs = self.n_fft // 2 + 1
        self.merged_freqs = self.freqs * self.audio_channels
        self.selected_bins = int(model.freq_indices.numel())
        self.pad = self.n_fft // 2
        self.padded_length = self.chunk_size + 2 * self.pad
        self.istft_length = (self.frames - 1) * self.hop_length + self.n_fft

        window = torch.hann_window(self.n_fft)
        freq = torch.arange(self.freqs, dtype=torch.float32).reshape(self.freqs, 1)
        time = torch.arange(self.n_fft, dtype=torch.float32).reshape(1, self.n_fft)
        angles = 2 * torch.pi * freq * time / self.n_fft

        self.register_buffer("stft_real_weight", (window * torch.cos(angles)).reshape(self.freqs, 1, self.n_fft))
        self.register_buffer("stft_imag_weight", (-window * torch.sin(angles)).reshape(self.freqs, 1, self.n_fft))

        scale = torch.full((self.freqs, 1), 2.0 / self.n_fft, dtype=torch.float32)
        scale[0, 0] = 1.0 / self.n_fft
        scale[-1, 0] = 1.0 / self.n_fft
        istft_real = window * scale * torch.cos(angles)
        istft_imag = -window * scale * torch.sin(angles)
        istft_weight = torch.cat([istft_real, istft_imag], dim=0).reshape(self.freqs * 2, 1, self.n_fft)
        self.register_buffer("istft_weight", istft_weight)

        envelope = torch.zeros(self.istft_length, dtype=torch.float32)
        window_square = window.square()
        for frame in range(self.frames):
            start = frame * self.hop_length
            envelope[start:start + self.n_fft] += window_square
        envelope = envelope.clamp_min(1e-11).reshape(1, 1, self.istft_length)
        self.register_buffer("istft_envelope", envelope)

        freq_indices = model.freq_indices.cpu().long()
        select_matrix = torch.zeros(self.merged_freqs, self.selected_bins, dtype=torch.float32)
        select_matrix[freq_indices, torch.arange(self.selected_bins)] = 1.0

        denom = torch.repeat_interleave(model.num_bands_per_freq.cpu().float(), self.audio_channels)
        reduce_matrix = select_matrix.transpose(0, 1) / denom.reshape(1, self.merged_freqs).clamp_min(1e-8)
        self.register_buffer("select_matrix", select_matrix)
        self.register_buffer("reduce_matrix", reduce_matrix)

    def _reflect_center_pad(self, audio: torch.Tensor) -> torch.Tensor:
        left = audio[:, :, 1:self.pad + 1].flip(dims=[-1])
        right = audio[:, :, self.chunk_size - self.pad - 1:self.chunk_size - 1].flip(dims=[-1])
        return torch.cat((left, audio, right), dim=-1)

    def _stft(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded = self._reflect_center_pad(audio)
        channel_audio = padded.reshape(self.audio_channels, 1, self.padded_length)
        real = F.conv1d(channel_audio, self.stft_real_weight, stride=self.hop_length)
        imag = F.conv1d(channel_audio, self.stft_imag_weight, stride=self.hop_length)

        real = real.reshape(1, self.audio_channels, self.freqs, self.frames)
        imag = imag.reshape(1, self.audio_channels, self.freqs, self.frames)
        real = real.permute(0, 2, 1, 3).reshape(1, self.merged_freqs, self.frames)
        imag = imag.permute(0, 2, 1, 3).reshape(1, self.merged_freqs, self.frames)

        real_bt = real.permute(0, 2, 1)
        imag_bt = imag.permute(0, 2, 1)
        selected_real = torch.matmul(real_bt, self.select_matrix)
        selected_imag = torch.matmul(imag_bt, self.select_matrix)
        packed = torch.stack((selected_real, selected_imag), dim=-1).reshape(
            1, self.frames, self.selected_bins * 2
        )
        return packed, real, imag

    def _apply_masks(
        self,
        packed_masks: torch.Tensor,
        stft_real: torch.Tensor,
        stft_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = packed_masks.reshape(1, self.frames, self.selected_bins, 2)
        mask_real = torch.matmul(masks[:, :, :, 0], self.reduce_matrix).permute(0, 2, 1)
        mask_imag = torch.matmul(masks[:, :, :, 1], self.reduce_matrix).permute(0, 2, 1)

        vocals_real = (stft_real * mask_real) - (stft_imag * mask_imag)
        vocals_imag = (stft_real * mask_imag) + (stft_imag * mask_real)
        return vocals_real, vocals_imag

    def _istft(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        real = real.reshape(1, self.freqs, self.audio_channels, self.frames)
        imag = imag.reshape(1, self.freqs, self.audio_channels, self.frames)
        real = real.permute(0, 2, 1, 3).reshape(self.audio_channels, self.freqs, self.frames)
        imag = imag.permute(0, 2, 1, 3).reshape(self.audio_channels, self.freqs, self.frames)

        stft = torch.cat((real, imag), dim=1)
        audio = F.conv_transpose1d(stft, self.istft_weight, stride=self.hop_length)
        audio = audio / self.istft_envelope
        audio = audio[:, :, self.pad:self.pad + self.chunk_size]
        return audio.reshape(1, self.audio_channels, self.chunk_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        packed_stft, stft_real, stft_imag = self._stft(audio)
        packed_masks = self.roformer_core(packed_stft)
        vocals_real, vocals_imag = self._apply_masks(packed_masks, stft_real, stft_imag)
        return self._istft(vocals_real, vocals_imag)


def replace_rotary_embeddings(model: nn.Module, frames: int) -> None:
    bands = len(model.band_split.dim_inputs)
    for time_transformer, freq_transformer in model.layers:
        for transformer_layer in time_transformer.layers:
            attention = transformer_layer[0]
            attention.rotary_embed = FixedRotaryEmbedding(
                attention.rotary_embed.freqs,
                outer_batch=bands,
                heads=attention.heads,
                seq_len=frames,
                dim=attention.to_qkv.out_features // (3 * attention.heads),
            )
            transformer_layer[0] = FixedAttention(attention, outer_batch=bands, seq_len=frames)

        for transformer_layer in freq_transformer.layers:
            attention = transformer_layer[0]
            attention.rotary_embed = FixedRotaryEmbedding(
                attention.rotary_embed.freqs,
                outer_batch=frames,
                heads=attention.heads,
                seq_len=bands,
                dim=attention.to_qkv.out_features // (3 * attention.heads),
            )
            transformer_layer[0] = FixedAttention(attention, outer_batch=frames, seq_len=bands)
