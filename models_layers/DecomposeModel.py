import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Portions of the code are inspired by sources including: https://github.com/luodhhh/ModernTCN/tree/main/ModernTCN-classification



# B: Batch size
# M: Number of variables in the multivariate sequence
# L: Sequence length (number of time steps)
# T: Number of time steps after embedding, can also be understood as the number of patches after splitting
# D: Number of channels per variable (embedding dimension)
# P: Kernel size of the embedding layer
# S: Stride of the embedding layer


# Modality-Specific Embedding (MSE) of Modality-Aware Decomposition
class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=2048):
        super(Embedding, self).__init__()
        self.P = P  # Kernel size for the convolutional layer
        self.S = S  # Stride for the convolutional layer
        self.conv = nn.Conv1d(
            in_channels=1,  # Input channel (1 for each variable)
            out_channels=D,  # Output channel (embedding dimension)
            kernel_size=P,  # Kernel size for the convolution operation
            stride=S  # Stride for the convolution operation
        )

    def forward(self, x):
        B = x.shape[0]  # Batch size, x: [B, M, L]
        x = x.unsqueeze(2)  # Add a channel dimension: [B, M, L] -> [B, M, 1, L]
        x = rearrange(x, 'b m r l -> (b m) r l')  # Merge batch and variable dimensions: [B, M, 1, L] -> [B*M, 1, L]

        # Padding for the convolution operation to maintain the sequence length after convolution
        x_pad = F.pad(
            x,
            pad=(0, self.P - self.S),  # Pad the sequence length to accommodate the convolution operation
            mode='replicate'  # Replicate the last elements for padding
        )  # [B*M, 1, L] -> [B*M, 1, L+P-S]

        x_emb = self.conv(x_pad)  # Apply the convolution operation: [B*M, 1, L+P-S] -> [B*M, D, T]
        x_emb = rearrange(x_emb, '(b m) d t -> b m d t', b=B)  # Reshape back to [B, M, D, T]

        return x_emb  # Output: [B, M, D, T]


# Cross-Channel Fusion (CCF) and Cross-Variable Fusion (CVF) via Point-Wise Convolution
class PWConv(nn.Module):
    def __init__(self, M, D, r, one=True):
        # one=True: Cross-Channel Fusion (CCF), one=False: Cross-Variable Fusion (CVF)
        super(PWConv, self).__init__()
        groups_num = M if one else D  # Determine groups for depthwise convolution based on fusion type

        # Cross-Channel Fusion (CCF) of Hierarchical Interaction Fusion
        self.pw_con1 = nn.Conv1d(
            in_channels=M * D,  # Input channels: M variables * D embedding dimension
            out_channels=r * M * D,  # Reduced output channels: r factor
            kernel_size=1,  # Point-wise convolution (kernel size of 1)
            groups=groups_num  # Group convolution (depthwise) to maintain specific channel/variable interaction
        )

        # Cross-Variable Fusion (CVF) of Hierarchical Interaction Fusion
        self.pw_con2 = nn.Conv1d(
            in_channels=r * M * D,  # Input channels from the previous layer
            out_channels=M * D,  # Output channels, restoring the original dimension
            kernel_size=1,  # Point-wise convolution (kernel size of 1)
            groups=groups_num  # Group convolution (depthwise) to maintain specific channel/variable interaction
        )

    def forward(self, x):
        # Apply two consecutive point-wise convolution operations with GELU activation
        x = self.pw_con2(F.gelu(self.pw_con1(x)))  # x: [B, M*D, T] -> [B, M*D, T]
        return x  # Output: [B, M*D, T]


# Decomposition Convolutional Block for feature extraction
class DecomposeConvBlock(nn.Module):
    def __init__(self, M, D, kernel_size, r):
        super(DecomposeConvBlock, self).__init__()

        # Local Temporal Extraction (LTE) of Modality-Aware Decomposition
        self.dw_conv = nn.Conv1d(
            in_channels=M * D,  # Input channels: M variables * D embedding dimension
            out_channels=M * D,  # Output channels remain the same (depthwise convolution)
            kernel_size=kernel_size,  # Kernel size for the convolution operation
            groups=M * D,  # Depthwise convolution for each variable-channel pair
            padding='same'  # Same padding to maintain the sequence length
        )
        self.bn = nn.BatchNorm1d(M * D)  # Batch normalization to stabilize training
        self.conv_ffn1 = PWConv(M, D, r, one=True)  # Cross-Channel Fusion (CCF) layer
        self.conv_ffn2 = PWConv(M, D, r, one=False)  # Cross-Variable Fusion (CVF) layer

    def forward(self, x_emb):
        D = x_emb.shape[-2]  # Extract embedding dimension from input x_emb: [B, M, D, T]

        # Reshape input for depthwise convolution
        x = rearrange(x_emb, 'b m d t -> b (m d) t')  # [B, M, D, T] -> [B, M*D, T]

        # Local Temporal Extraction (LTE) with depthwise convolution
        x = self.dw_conv(x)  # [B, M*D, T] -> [B, M*D, T]
        x = self.bn(x)  # Apply batch normalization: [B, M*D, T] -> [B, M*D, T]
        x = self.conv_ffn1(x)  # Cross-Channel Fusion (CCF): [B, M*D, T] -> [B, M*D, T]

        # Reshape and permute to apply Cross-Variable Fusion (CVF)
        x = rearrange(x, 'b (m d) t -> b m d t', d=D)  # [B, M*D, T] -> [B, M, D, T]
        x = x.permute(0, 2, 1, 3)  # Permute to [B, D, M, T]
        x = rearrange(x, 'b d m t -> b (d m) t')  # Flatten to [B, D*M, T]

        # Cross-Variable Fusion (CVF)
        x = self.conv_ffn2(x)  # [B, D*M, T] -> [B, D*M, T]

        # Reshape back to the original shape
        x = rearrange(x, 'b (d m) t -> b d m t', d=D)  # [B, D*M, T] -> [B, D, M, T]
        x = x.permute(0, 2, 1, 3)  # Permute back to [B, M, D, T]

        # Add residual connection
        out = x + x_emb  # Residual connection for the output

        return out  # Final output: [B, M, D, T]
