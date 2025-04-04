import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding,DataEmbedding_wo_val
from layers.Conv_Blocks import Inception_Block_V1

"""TimesNet_v2: 将最后的 adaptive aggregation 换为 SE 通道注意力机制"""




class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_len, channel, k = x.size()

        # Global Average Pooling on seq_len and channel dimensions
        y = self.avg_pool(x.permute(0, 3, 1, 2)).view(batch_size, k)  # y shape: [batch_size, k]  # y shape: [batch_size, k]

        # Fully connected layers
        y = self.fc1(y)  # y shape: [batch_size, channel // reduction]
        y = F.relu(y)
        y = self.fc2(y)  # y shape: [batch_size, channel]
        y = torch.sigmoid(y).unsqueeze(1).unsqueeze(1)  # y shape: [batch_size, 1, 1, k]

        # Apply the scaling to the input tensor
        return (x * y).sum(-1)

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        self.se = SqueezeExcitation(self.k)
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)
        # apply SE block
        res = self.se(res)  # apply SE attention to res

        # # adaptive aggregation with learned attention weights
        # period_weight = F.softmax(res.mean(dim=1), dim=-1).unsqueeze(1).unsqueeze(1)
        #
        # # Ensure res and period_weight shapes match for multiplication
        # res = torch.sum(res * period_weight.repeat(1, T, N, 1), dim=-1)




        # res = torch.stack(res, dim=-1)
        # # adaptive aggregation
        # period_weight = F.softmax(period_weight, dim=1)
        # period_weight = period_weight.unsqueeze(
        #     1).unsqueeze(1).repeat(1, T, N, 1)
        # res = torch.sum(res * period_weight, -1)


        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        # self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)



    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        # output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        feature_output = output.reshape(output.shape[0], -1)
        output = self.projection(feature_output)  # (batch_size, num_classes)
        return feature_output,output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        if self.task_name == 'classification':
            enc_out,dec_out = self.classification(x_enc, x_mark_enc)
            return enc_out,dec_out  # [B, N]
        return None
