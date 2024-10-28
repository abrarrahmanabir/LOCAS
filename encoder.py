import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.norm(out)
        out = self.relu(out)
        return out

class CNNAttention(nn.Module):
    def __init__(self, output_dim):
        super(CNNAttention, self).__init__()

        self.resconv1 = ResidualConvBlock(1, 64, 3)
        self.resconv2 = ResidualConvBlock(64, 128, 3)
        self.resconv3 = ResidualConvBlock(128, 256, 3)

        self.attention1 = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        self.fc = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape to [batch_size, 1, embedding_dim]
        x = self.resconv1(x)
        x = self.resconv2(x)
        x = self.resconv3(x)

        x = x.permute(0, 2, 1)  # Adjust shape for attention

        attn_output1, _ = self.attention1(x, x, x)
        # Apply residual connection around attention
        attn_output1 += x
        attn_output1 = self.relu(attn_output1)

        attn_output2, _ = self.attention2(attn_output1, attn_output1, attn_output1)
        # Second residual connection
        attn_output2 += attn_output1
        attn_output2 = self.relu(attn_output2)

        # Pool the attention output to get fixed size output
        pooled = F.adaptive_avg_pool1d(attn_output2.permute(0, 2, 1), 1).squeeze(2)

        out = self.fc(pooled)
        return out



class SupConCNNAttention(nn.Module):
    """CNNAttention + projection head"""
    def __init__(self, output_dim, projection_dim=64):
        super(SupConCNNAttention, self).__init__()

        # Initialize the CNNAttention as the encoder
        self.encoder = CNNAttention(output_dim)

        dim_in = output_dim

        self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),  # First layer of MLP
                nn.ReLU(inplace=True),  # Activation function
                nn.Linear(dim_in, projection_dim)  # Output layer of MLP
            )

    def forward(self, x):
        # Pass the input through the encoder
        feat = self.encoder(x)

        # Pass the output of the encoder through the projection head
        # And normalize the output feature vector
        feat = F.normalize(self.head(feat), dim=1)

        return feat