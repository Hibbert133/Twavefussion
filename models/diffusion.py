import math
import torch
import torch.backends
import torch.backends.cudnn
import torch.nn as nn
import os
import yaml
import argparse
import torch.nn.functional as F
# torch.backends.cudnn.enable = True
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False

# os.environ['CUDA_LAUNCH_BLOCKING']='1'

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1
                                       )

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1)

    def forward(self, x):
        if self.with_conv:
            #x = self.conv(x)
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=1
                                     )
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1)

    def forward(self, x, temb):
        h = x

        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"

        # 对输入做标准化处理
        self.norm = Normalize(in_channels)

        # Query, Key, Value 的线性变换
        self.q = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.k = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.v = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)

        # 输出投影
        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 标准化输入
        h_ = self.norm(x)

        # 获取 q, k, v 向量
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 获取 batch size, 通道数（即总的维度数），长度
        b, c, h = q.shape

        # 将 q, k, v 分成多个头，维度调整为 (batch, num_heads, head_dim, length)
        q = q.view(b, self.num_heads, self.head_dim, h)
        k = k.view(b, self.num_heads, self.head_dim, h)
        v = v.view(b, self.num_heads, self.head_dim, h)

        # 转置 q 和 k，维度调整为 (batch, num_heads, length, head_dim)
        q = q.permute(0, 1, 3, 2)  # (b, num_heads, h, head_dim)
        k = k.permute(0, 1, 2, 3)  # (b, num_heads, head_dim, h)

        # 计算注意力权重
        w_ = torch.matmul(q, k) * (self.head_dim ** -0.5)  # (b, num_heads, h, h)
        w_ = torch.nn.functional.softmax(w_, dim=-1)  # 在最后一个维度上进行 softmax

        # 应用注意力权重到 v
        v = v.permute(0, 1, 3, 2)  # (b, num_heads, h, head_dim)
        attn_output = torch.matmul(w_, v)  # (b, num_heads, h, head_dim)
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous()  # (b, num_heads, head_dim, h)

        # 将多个头的输出拼接起来 (batch, in_channels, length)
        attn_output = attn_output.view(b, c, h)

        # 输出投影
        h_ = self.proj_out(attn_output)

        # 跳跃连接
        return x + h_

# class AttnBlock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels

#         self.norm = Normalize(in_channels)
#         self.q = torch.nn.Conv1d(in_channels,
#                                  in_channels,
#                                  kernel_size=1)
#         self.k = torch.nn.Conv1d(in_channels,
#                                  in_channels,
#                                  kernel_size=1)
#         self.v = torch.nn.Conv1d(in_channels,
#                                  in_channels,
#                                  kernel_size=1)
#         self.proj_out = torch.nn.Conv1d(in_channels,
#                                         in_channels,
#                                         kernel_size=1)

#     def forward(self, x):
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         # compute attention
#         b, c, h = q.shape
#         q = q.reshape(b, c, h)
#         q = q.permute(0, 2, 1)   # b,h,c
#         k = k.reshape(b, c, h)  # b,c,h
#         w_ = torch.bmm(q, k)     # b,h,h   w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
#         w_ = w_ * (int(c)**(-0.5))
#         w_ = torch.nn.functional.softmax(w_, dim=2)

#         # attend to values
#         v = v.reshape(b, c, h)
#         w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
#         # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
#         h_ = torch.bmm(v, w_)
#         h_ = h_.reshape(b, c, h)

#         h_ = self.proj_out(h_)

#         return x+h_


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.model.input_size#32
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)

        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),  # 第一层
            # torch.nn.Linear(self.temb_ch, self.temb_ch * 2),  # 扩展为两倍
            torch.nn.Linear(self.temb_ch , self.temb_ch),  # 再次映射回原始维度
        ])
        
        # downsampling

        self.conv_in = nn.Conv1d(out_ch,ch,1)

        curr_res = resolution

        in_ch_mult = (1,)+ch_mult
        self.attn_in_layers_down = []  # 初始化下采样的 attn_in_layers_down
        self.attn_in_layers_up = []    # 初始化上采样的 attn_in_layers_up


        self.down = nn.ModuleList()
        # self.attn_in_layers = []
        block_in = None
        for i_level in range(self.num_resolutions):
            attn_in_blocks = []
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                has_attn = curr_res in attn_resolutions
                attn_in_blocks.append(has_attn)  # 记录当前块是否有注意力模块
                if has_attn:
                    attn.append(AttnBlock(block_in))
                    print(f"Down Added AttnBlock at resolution {curr_res}, level {i_level}, block {i_block}")
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            self.attn_in_layers_down.append(attn_in_blocks)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_2 = AttnBlock(block_in)  # 第二次注意力
        self.mid.block_3 = ResnetBlock(in_channels=block_in,  # 可选的额外 ResNet 块
                               out_channels=block_in,
                               temb_channels=self.temb_ch,
                               dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        #self.attn_in_layers_up = []
        curr_res = curr_res 
        for i_level in reversed(range(self.num_resolutions)):
            attn_in_blocks = []
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]


                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                has_attn = curr_res in attn_resolutions
                attn_in_blocks.append(has_attn)
                if has_attn:
                    attn.append(AttnBlock(block_in))
                    #print(f"Up Added AttnBlock at resolution {curr_res}, level {i_level}, block {i_block}")
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
            #self.attn_in_layers_up.insert(0, attn_in_blocks)
            self.attn_in_layers_up.insert(0, attn_in_blocks)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1)
###########################################################################################################
    def forward(self, x, t):


        # timestep embedding

        temb = get_timestep_embedding(t, self.ch)
        #128*128
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        # temb = torch.nn.SiLU()(temb)
        # temb = self.temb.dense[2](temb)


        # downsampling
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            attn_idx = 0
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # if len(self.down[i_level].attn) > 0:
                #     h = self.down[i_level].attn[i_block](h)
                if self.attn_in_layers_down[i_level][i_block]:
                    #print(f"Down Applying AttnBlock at level {i_level}, block {i_block}")
                    h = self.down[i_level].attn[attn_idx](h)
                    attn_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.mid.attn_2(h)
        h = self.mid.block_3(h, temb)

        #print("attn_in_layers_up before upsampling:", self.attn_in_layers_up)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            attn_idx = 0
            for i_block in range(self.num_res_blocks+1):

                hspop = hs.pop()

                h = self.up[i_level].block[i_block](
                    torch.cat([h, hspop], dim=1), temb)

                # if len(self.up[i_level].attn) > 0:

                #     h = self.up[i_level].attn[i_block](h)
                if self.attn_in_layers_up[i_level][i_block]:
                    #print(f"Up Applying AttnBlock in upsampling at level {i_level}, block {i_block}")
                    h = self.up[i_level].attn[attn_idx](h)
                    attn_idx += 1
            if i_level != 0:

                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h
    


class Model2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.model.input_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # self.attn_in_down_layers = []#############
        # self.attn_in_up_layers = []############

        # Timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        self.attn_in_layers_down = []  # 初始化下采样的 attn_in_layers_down
        self.attn_in_layers_up = []
        # Input layer
        self.conv_in = nn.Conv1d(out_ch, ch, 1)
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult

        # Downsampling
        self.down = nn.ModuleList()
        self.skip_connections = nn.ModuleDict()  # Used for dense connections

        block_in = None
        #self.attn_in_layers = []########################################
        for i_level in range(self.num_resolutions):
            attn_in_blocks = []
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                has_attn = curr_res in attn_resolutions
                attn_in_blocks.append(has_attn)
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
                #     attn_in_blocks.append(True)

                # else:
                #     attn_in_blocks.append(False)
                if has_attn:
                    attn.append(AttnBlock(block_in))
                    #print(f"Down Added AttnBlock at resolution {curr_res}, level {i_level}, block {i_block}")
            #self.attn_in_down_layers.append(attn_in_blocks)
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            self.attn_in_layers_down.append(attn_in_blocks)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            attn_in_blocks = []
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
                #     attn_in_blocks.append(True)
                # else:
                #     attn_in_blocks.append(False)
                has_attn = curr_res in attn_resolutions
                attn_in_blocks.append(has_attn)
                if has_attn:
                    attn.append(AttnBlock(block_in))
                    #print(f"Up Added AttnBlock at resolution {curr_res}, level {i_level}, block {i_block}")
            #self.attn_in_up_layers.append(attn_in_blocks)
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
            self.attn_in_layers_up.insert(0, attn_in_blocks)

        # Output layer
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, out_ch, kernel_size=1)

    def forward(self, x, t):
        # Timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            attn_idx = 0
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # if len(self.down[i_level].attn) > 0:
                #     h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
                if self.attn_in_layers_down[i_level][i_block]:  # Check if attention is applied
                    #print(f"Down Applying AttnBlock at level {i_level}, block {i_block}")
                    h = self.down[i_level].attn[attn_idx](h)
                    attn_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling with dense connections
        for i_level in reversed(range(self.num_resolutions)):
            attn_idx = 0
            for i_block in range(self.num_res_blocks + 1):
                hspop = hs.pop()
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hspop], dim=1), temb)
                # if len(self.up[i_level].attn) > 0:
                #     h = self.up[i_level].attn[i_block](h)
                if self.attn_in_layers_up[i_level][i_block]:  # Check if attention is applied
                    h = self.up[i_level].attn[attn_idx](h)
                    attn_idx += 1
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h



if __name__ == '__main__':
    with open('../configs/test.yml', "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    device = torch.device("cpu")

    new_config.device = device

    model = Model(new_config)
    model = model.to(device)

    #data = torch.randn(128,3,32,32)
    for i in range(10):
        data = torch.randn(128, 64, 38)
        t = torch.randint(1000, size=(data.shape[0],))
        data = model(data, t)
        print(data.shape)





    print('test')