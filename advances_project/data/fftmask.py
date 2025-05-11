import math
import torch
import torch.nn as nn


class FFTMask(nn.Module):
    def __init__(self, length=64):
        super().__init__()

        assert math.log2(length).is_integer(), "length must be a power of 2"

        self.length = length
        self.F = length // 2 + 1  # rfft length
        self.G = int(math.log2(length))

        # m represents fft bins in each group
        m = torch.zeros(self.G, self.F)
        splits = 2 ** torch.arange(0, self.G)
        for i in range(self.G):
            if i == 0:
                m[i, 0:1] = 1
            elif i == 1:
                m[i, 1:3] = 1
            else:
                m[i, splits[i] // 2 + 1 : splits[i] + 1] = 1

        self.register_buffer("m", m)

        # Assign normalized frequencies to bins
        self.register_buffer("c", torch.linspace(0, 1, self.F).unsqueeze(0))

    def forward(self, x, provided_group_mask=None):
        assert x.ndim == 4, "x must have 4 dimensions"
        assert x.shape[-1] == self.length, "input length must match FFT Length"

        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype

        if provided_group_mask is None:
            thresholds = torch.rand(batch_size, 1, device=device, dtype=dtype)
            scores = torch.rand(batch_size, self.G, device=device, dtype=dtype)
            fft_mask = (scores > thresholds).to(dtype) @ self.m
        else:
            fft_mask = provided_group_mask.to(dtype) @ self.m

        fft_mask = fft_mask[:, :, None] * fft_mask[:, None, :]
        top_right = fft_mask[:, :, 1:-1].flip(-1)
        bottom_left = fft_mask[:, 1:-1, :].flip(-2)
        bottom_right = fft_mask[:, 1:-1, 1:-1].flip(-1, -2)

        top = torch.cat((fft_mask, top_right), dim=-1)
        bottom = torch.cat([bottom_left, bottom_right], dim=-1)
        full_fft_mask = torch.cat([top, bottom], dim=-2)
        return torch.real(torch.fft.ifft2(full_fft_mask * torch.fft.fft2(x)))
