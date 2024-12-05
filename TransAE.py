import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(AttentionEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, 16)  #  [batch_size, seq_length, 64]

    def forward(self, x):
        #  x[batch_size, seq_length, input_dim]
        x = self.embedding(x)  # ->embed_dim
        x = x.permute(1, 0, 2)
        x_enc = self.encoder(x) 
        x_enc = x_enc.permute(1, 0, 2)  #  [batch_size, seq_length, embed_dim]
        x_enc = self.output_layer(x_enc)  #  [batch_size, seq_length, 64]
        return x_enc

class AttentionDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, ff_dim, dropout=0.1, use_act=True):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, ff_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=ff_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(ff_dim, output_dim)  # 调整输出为 [batch_size, seq_length, output_dim]
        self.use_act = use_act
        self.act = nn.Sigmoid() if use_act else nn.Identity()

    def forward(self, z, memory):
        # z [batch_size, seq_length, input_dim]
        z = self.embedding(z)  #z ->ff_dim
        # z = z.permute(1, 0, 2)  # [seq_length, batch_size, ff_dim]
        z = z.permute(1, 0, 2)

        memory = self.embedding(memory)  # Ensure that the dimensions of memory are the same as z
        memory = memory.permute(1, 0, 2)  #  [seq_length, batch_size, ff_dim]
        z_dec = self.decoder(z, memory)  
        z_dec = z_dec.permute(1, 0, 2)  #  [batch_size, seq_length, ff_dim]
        z_dec = self.output_layer(z_dec)  #  [batch_size, seq_length, output_dim]
        z_dec = self.act(z_dec)  
        return z_dec



# Attention-based Encoder-Decoder Auto-Encoder
class TransformerAE(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, dropout=0.2, use_act=True):
        super(TransformerAE, self).__init__()
        self.encoder = AttentionEncoder(input_dim, embed_dim, num_heads, num_layers, ff_dim, dropout)
        self.decoder = AttentionDecoder(16, input_dim, num_heads, num_layers, ff_dim, dropout, use_act)

    def forward(self, x, flag):
        #  x ：[batch_size, window_size, feature_dim]
        x = nn.ReLU()(x) 
        
        # 编码过程
        if flag == 'all':
            x_enc = self.encoder(x)
            x_dec = self.decoder(x_enc, x_enc)
            return x_dec
        elif flag == 'en':
            x_enc = self.encoder(x)
            return x_enc
        elif flag == 'de':
            x_dec = self.decoder(x, x) 
            return x_dec

# # # 使用示例
# input_dim = 25   # 输入特征维度
# embed_dim = 128  # 嵌入维度
# num_heads = 4    # 多头注意力的头数
# num_layers = 3   # 编码器/解码器的层数
# ff_dim = 256     # 前馈神经网络的隐藏层维度

# # 初始化模型
# model = TransformerAE(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads,
#                       num_layers=num_layers, ff_dim=ff_dim)

# # 示例输入，形状为 [batch_size, seq_length, input_dim]
# x = torch.randn(128, 64, 25)  # [128, 64, 25]

# # 前向传播
# output = model(x, flag='all')

# # 查看输出形状
# print("Output shape:", output.shape)  # 输出应为 [128, 64, 25]

