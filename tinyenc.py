import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as param
import torch.nn.utils.weight_norm as weight_norm
import warnings
import numpy as np
import typing as tp

# Supported normalizations
CONV_NORMALIZATIONS = {'none', 'weight_norm', 'spectral_norm', 'layer_norm', 'time_group_norm'}

# -- Helper functions --

def pad1d(x, pad, mode='reflect'):
    return nn.functional.pad(x, pad, mode=mode)

def get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total):
    T = x.shape[-1]
    output_len = (T + padding_total - kernel_size) // stride + 1
    needed_len = (output_len - 1) * stride + kernel_size - padding_total
    extra = needed_len - T
    return max(extra, 0)

# -- Normalization wrapper functions --

def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm(module)
    else:
        return module

def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    if norm == 'layer_norm':
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

class ConvLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # Convert [B, C, T] -> [B, T, C] for LayerNorm, then back
        return self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)

# -- Building blocks --

class NormConv1d(nn.Module):
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.conv(x))

class SConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True, causal=False, norm='none',
                 norm_kwargs={}, pad_mode='reflect'):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn("SConv1d with stride > 1 and dilation > 1")
        self.conv = NormConv1d(in_channels, out_channels, kernel_size,
                               stride=stride, dilation=dilation, groups=groups,
                               bias=bias, causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        effective_kernel = (kernel_size - 1) * dilation + 1
        padding_total = effective_kernel - stride
        extra = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            x = pad1d(x, (padding_total, extra), mode=self.pad_mode)
        else:
            pad_right = padding_total // 2
            pad_left = padding_total - pad_right
            x = pad1d(x, (pad_left, pad_right + extra), mode=self.pad_mode)
        return self.conv(x)

class SEANetResnetBlock(nn.Module):
    def __init__(self, dim, kernel_sizes, dilations, norm, norm_params,
                 activation, activation_params, causal, pad_mode, compress=2, true_skip=False):
        super().__init__()
        self.true_skip = true_skip
        act = getattr(nn, activation)

        mid_dim = dim // compress

        self.block = nn.Sequential(
            act(**activation_params),
            SConv1d(dim, mid_dim, kernel_sizes[0], dilation=dilations[0], norm=norm,
                    norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
            act(**activation_params),
            SConv1d(mid_dim, dim, kernel_sizes[1], dilation=dilations[1], norm=norm,
                    norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
        )
        if true_skip:
            self.skip = SConv1d(dim, dim, 1, norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)

class SLSTM(nn.Module):
    def __init__(self, dimension, num_layers=1, skip=True):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, num_layers)
        self.skip = skip

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        return y.permute(1, 2, 0)

# -- Tiny SEANet Encoder --

class SEANetEncoder(nn.Module):
    def __init__(self, channels=1, dimension=128, n_filters=32, n_residual_layers=1,
                 ratios=[8, 5, 4, 2], activation='ELU', activation_params={'alpha': 1.0},
                 norm='weight_norm', norm_params={}, kernel_size=7,
                 last_kernel_size=7, residual_kernel_size=3, dilation_base=2,
                 causal=False, pad_mode='reflect', true_skip=False, compress=2, lstm=2, codebook_size=1024):
        super().__init__()
        self.hop_length = np.prod(ratios)
        self.ratios = list(reversed(ratios))
        self.codebook_size = codebook_size

        act = getattr(nn, activation)
        mult = 1
        model = [
            SConv1d(channels, mult * n_filters, kernel_size, norm=norm,
                    norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        for i, ratio in enumerate(self.ratios):
            for j in range(n_residual_layers):
                model += [SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                            dilations=[dilation_base ** j, 1], norm=norm,
                                            norm_params=norm_params, activation=activation,
                                            activation_params=activation_params, causal=causal,
                                            pad_mode=pad_mode, compress=compress, true_skip=true_skip)]
            model += [
                act(**activation_params),
                SConv1d(mult * n_filters, mult * n_filters * 2,
                        kernel_size=ratio * 2, stride=ratio, norm=norm,
                        norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
            ]
            mult *= 2

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            SConv1d(mult * n_filters, dimension, last_kernel_size,
                    norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

        self.proj = nn.Conv1d(dimension, 2, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        x = self.proj(x)
        
        # During training use straight-through estimator
        if self.training:
            x_hard = torch.round(torch.sigmoid(x) * (self.codebook_size - 1))
            # Straight-through trick: forward is discretized but backward is continuous
            x = x_hard.detach() - torch.sigmoid(x).detach() + torch.sigmoid(x)
        else:
            x = torch.round(torch.sigmoid(x) * (self.codebook_size - 1))
        return x

class TinySEANetEncoder(SEANetEncoder):
    def __init__(self):
        super().__init__(
            channels=1,
            dimension=128,
            n_filters=16,
            n_residual_layers=1,
            ratios=[8, 5, 8],
            activation='ELU',
            activation_params={'alpha': 1.0},
            norm='weight_norm',
            norm_params={},
            kernel_size=7,
            last_kernel_size=5,
            residual_kernel_size=3,
            dilation_base=2,
            causal=False,
            pad_mode='reflect',
            true_skip=False,
            compress=4,
            lstm=1,
            codebook_size=1024
        )

