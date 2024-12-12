import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model*configs.window_size, configs.num_class
            )

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc,None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        feature_output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        # feature_output=output.mean(dim=1)
        output = self.projection(feature_output)  # (batch_size, num_classes)
        return feature_output,output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        if self.task_name == "classification":
            enc_out,dec_out = self.classification(x_enc, x_mark_enc)
            return enc_out,dec_out  # [B, N]
        return None



