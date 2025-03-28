import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_attention import TemporalAttention, SelfAttention
from models_layers.DecomposeModel import *
from layers.Mamba_Family import Mamba_Layer, AM_Layer, Att_Layer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
from mamba_ssm import Mamba
from einops import rearrange


# B: batch size
# M: number of variables in the multivariate sequence
# L: length of the sequence (number of time steps)
# T: number of time steps after embedding, also can be considered as the number of patches after splitting
# D: number of channels per variable
# P: kernel size of the embedding layer
# S: stride of the embedding layer
class DecomposeWHAR(nn.Module):
    def __init__(self,
                 num_sensor,  # Number of sensors (N)
                 M,  # Number of variables in the multivariate sequence
                 L,  # Length of the input sequence (time steps)
                 D=64,  # Number of channels per variable
                 P=8,  # Kernel size of the embedding layer
                 S=4,  # Stride of the embedding layer
                 kernel_size=5,  # Kernel size for convolutional layers
                 r=1,  # A hyperparameter for decomposition (e.g., reduction ratio)
                 num_layers=2,  # Number of decomposition layers
                 num_m_layers=1,   # Number of mamba layers
                 num_a_layers=1,   # Number of attention layers
                 num_classes=17):  # Number of classes for classification
        super(DecomposeWHAR, self).__init__()
        self.num_layers = num_layers
        self.num_a_layers = num_a_layers
        self.num_m_layers = num_m_layers
        T = L // S  # Calculate the number of patches after embedding
        self.T = T
        self.D = D
        # Embedding layer to transform input sequences into higher dimensional representations
        self.embed_layer = Embedding(P, S, D)
        # Backbone consisting of multiple decomposition convolutional blocks
        self.backbone = nn.ModuleList([DecomposeConvBlock(M, D, kernel_size, r) for _ in range(num_layers)])
        # Fully connected output layer for classification
        self.fc_out = nn.Linear(num_sensor * D * T, num_classes)
        self.dropout_prob = 0.6  # Dropout probability

        d_model = D * T  # Model dimension after embedding
        d_model_mamba = num_sensor * D  # Model dimension for Mamba preprocessing
        d_state = 16  # State dimension for Mamba
        d_conv = 4  # Convolutional dimension for Mamba
        dropout = 0.05  # Dropout for attention layers
        factor = 1  # Factor for the FullAttention layer
        n_heads = 8  # Number of attention heads
        self.d_layers = 1  # Number of attention layers

        # Mamba Block of Global Temporal Aggregation (GTA)
        self.mamba_preprocess = nn.ModuleList([
            Mamba_Layer(Mamba(d_model=d_model_mamba, d_state=d_state, d_conv=d_conv), d_model_mamba)
            for _ in range(num_m_layers)
        ])

        # Attention layers of Cross-Sensor Interaction (CSI)
        self.AM_layers = nn.ModuleList(
            [
                Att_Layer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=True),
                        d_model, n_heads),
                    d_model,
                    dropout
                )
                for i in range(self.d_layers)
            ]
        )
        self.register_buffer(
            "centers", (torch.randn(num_classes, num_sensor*D * T).cuda())
        )

    def forward(self, inputs):  # inputs: (B, N, L, M) - Batch size, Number of sensors, Sequence length, Number of variables
        
        B, N, L, M = inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3] # 64,5,24,9
        x = inputs.reshape(B*N, L, M)  # (B*N, L, M)
        x = x.permute(0, 2, 1)  # (B*N, M, L)


        x_emb = self.embed_layer(x)  # [B*N, M, L] -> [B*N, M, D, T]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B*N, M, D, L] -> [B*N, M, D, T]

        # Flatten
        x_emb = rearrange(x_emb, 'b m d t -> b m (d t)', b=B*N, m=M)  # [B*N, M, D, T] -> [B*N, M, D*T]

        # Aggregate over the sensor dimension and apply classification head
        x_emb = x_emb.mean(dim=1)  # [B*N, M, D*T] -> [B*N, D*T] 64*5,D*T
        x_emb = x_emb.reshape(inputs.shape[0],inputs.shape[1],-1) # B,N,D*T (64,5,D*T)



        # Reshape to B,T,N*D (64, T, 5*D)
        x_emb = x_emb.reshape(inputs.shape[0],inputs.shape[1],self.D,self.T) # B,N,D,T (64,5,D,T)
        x_emb = x_emb.permute(0,3,1,2)
        x_emb = x_emb.reshape(inputs.shape[0],x_emb.shape[1],-1) # B,T,N*D (64,T,5*D)




        # Mamba Input
        for i in range(0,self.num_m_layers):
            x_emb = self.mamba_preprocess[i](x_emb)  # B,T,N*D


        # Reshape to B, N, D*T
        x_emb = x_emb.reshape(inputs.shape[0], x_emb.shape[1], inputs.shape[1], self.D)  # B,T,N,D
        x_emb = x_emb.permute(0, 2, 3, 1)  # B,N,D,T
        x_emb = x_emb.reshape(inputs.shape[0], x_emb.shape[1], -1)  # B,N,D*T



        # Attention
        for i in range(self.num_a_layers):
            x_emb, output_attention = self.AM_layers[i](x_emb, None)  # B,N,D*T



        x = x_emb

        x = x.reshape(inputs.shape[0],-1)

        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        pred = self.fc_out(x)  # [B, D*T] -> [B, num_classes]

        return x,pred # output: [B, num_classes]
