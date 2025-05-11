import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, down=False, weight_init_scale=1.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.down = down
        self.weight_init_scale = weight_init_scale

        if down:
            resample_filter = torch.tensor([[1, 1], [1, 1]]) / 4
            resample_filter = resample_filter.expand(in_channels, 1, 2, 2)
            self.register_buffer("resample_filter", resample_filter, persistent=False)

        # Actual Convolution
        if kernel_size:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size // 2),
            )

            # Initialize Weights and Biases
            with torch.no_grad():
                self.conv.weight.mul_(self.weight_init_scale)
                self.conv.bias.mul_(0.0)

    def forward(self, x):
        if self.down:
            x = F.conv2d(x, self.resample_filter, stride=2, groups=self.in_channels)

        if self.kernel_size:
            x = self.conv(x)

        return x

    def __repr__(self):
        if self.kernel_size:
            return self.conv.__repr__()
        else:
            if self.down:
                return "DownResample()"
            else:
                return "Identity()"


class GroupNorm(nn.Module):
    def __init__(self, num_channels, target_num_groups=32, min_channels_per_group=4):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = min(target_num_groups, num_channels // min_channels_per_group)
        if self.num_groups == 0:
            raise ValueError("Num. channels less than min. channels per group")
        self.gn = nn.GroupNorm(self.num_groups, num_channels, eps=1e-06)

    def forward(self, x):
        return self.gn(x)

    def __repr__(self):
        return self.gn.__repr__()


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        down=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.down = down

        # Layers
        self.norm0 = GroupNorm(in_channels)
        self.conv0 = Conv2d(in_channels, out_channels, kernel_size, down=down)

        self.norm1 = GroupNorm(out_channels)
        self.conv1 = Conv2d(
            out_channels, out_channels, kernel_size, weight_init_scale=1e-5
        )

        # Skip
        if (in_channels != out_channels) or down:
            k = None if (in_channels == out_channels) else 1
            self.skip_proj = Conv2d(in_channels, out_channels, kernel_size=k, down=down)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        residual_stream = x
        x = self.conv0(F.silu(self.norm0(x)))
        x = self.conv1(F.silu(self.norm1(x)))
        x = (x + self.skip_proj(residual_stream)) / math.sqrt(2)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        data_resolution=512,
        in_channels=3,
        out_channels=4,
        dims=[32, 32, 64, 128],
        num_blocks_per_res=3,
        kernel_size=3,
    ):
        super().__init__()
        # Filling out Fields
        self.data_resolution = data_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = dims
        self.num_blocks_per_res = num_blocks_per_res
        self.kernel_size = kernel_size

        # Number of Levels
        self.num_levels = len(dims)

        self._build_encoder()

        # Print number of params
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"Encoder Number of Parameters: {self.num_params:,}")

    def forward(self, x):
        for _, module in self.enc.items():
            x = module(x)

        return x

    def _build_encoder(self):
        self.enc = torch.nn.ModuleDict()

        for level in range(self.num_levels):
            res = self.data_resolution >> level

            res_out_channels = self.dims[level]

            if level == 0:
                res_in_channels = self.dims[0]

                self.enc[f"{res}_conv0"] = Conv2d(
                    in_channels=self.in_channels,
                    out_channels=res_in_channels,
                    kernel_size=self.kernel_size,
                )

            else:
                res_in_channels = self.dims[level - 1]

                self.enc[f"{res * 2}->{res}_down"] = Block(
                    in_channels=res_in_channels,
                    out_channels=res_in_channels,
                    down=True,
                    kernel_size=self.kernel_size,
                )

            for block_idx in range(self.num_blocks_per_res):
                block_in_channels = (
                    res_in_channels if block_idx == 0 else res_out_channels
                )

                self.enc[f"{res}_block{block_idx}"] = Block(
                    in_channels=block_in_channels,
                    out_channels=res_out_channels,
                    kernel_size=self.kernel_size,
                )

        self.enc["outconv"] = Conv2d(
            in_channels=res_out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
        )
