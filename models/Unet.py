import torch
import torch.nn as nn

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups = 8, num_channels=in_channels, eps=1e-6, affine=True)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h = q.shape
        q = q.reshape(b, c, h)
        q = q.permute(0, 2, 1)   # b,h,c
        k = k.reshape(b, c, h)  # b,c,h
        w_ = torch.bmm(q, k)     # b,h,h   w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h)

        h_ = self.proj_out(h_)

        return x+h_


# Light UNet
class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.unet.ch, config.unet.out_ch, tuple(config.unet.ch_mult)
        num_res_blocks = config.unet.num_res_blocks
        dropout = config.unet.dropout
        in_channels = config.unet.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.unet.resamp_with_conv

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # 输入层
        self.conv_in = nn.Conv1d(out_ch, ch, kernel_size=1)
        curr_res = resolution #38
        in_ch_mult = (1,) + ch_mult

        # 下采样模块
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):#4
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):#2
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # 中间层
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_2 = AttnBlock(block_in)
        self.mid.block_3 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)


        # 上采样模块
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # 输出层
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, out_ch, kernel_size=1)

    def forward(self, x):
        # 下采样
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                # print(h.shape)
                # print(f"i_level and i_block: {i_level,i_block}")
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 中间层
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.mid.attn_2(h)
        h = self.mid.block_3(h)
        # print(f"h shape: {h.shape}")
        # print(f"hspop shape: {hs[-1].shape}")

        # 上采样
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1 ):
                hspop = hs.pop()
                # print(hspop.shape)
                # print(f"h shape: {h.shape}, hspop shape: {hspop.shape}")
                # # print(f"hspop shape: {hspop.shape}")
                # print(f"i_level and i_block: {i_level,i_block}")
                
                h = self.up[i_level].block[i_block](torch.cat([h, hspop], dim=1))
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 输出层
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        if self.with_conv:
            # x = self.conv(x)
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        else:
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # if self.with_conv:
        #     x = self.conv(x)
        return x
