
######################### ENCODER NETWORK ###################################

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
############### ML DECODER #########################


import torch
from typing import Optional
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        #self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) ###################################


        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, memory, memory)[0] ######################################################3


        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class MLDecoder(nn.Module):
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768,
                 initial_num_features=2048, zsl=0):
        super(MLDecoder, self).__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(initial_num_features, decoder_embedding)

        # non-learnable queries
        if not zsl:
            query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            query_embed.requires_grad_(False)
        else:
            query_embed = None

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048


        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)

        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(decoder_embedding, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None

    def forward(self, x):
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.decoder.query_embed))
        else:
            query_embed = self.decoder.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional


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

import torch
import torch.nn as nn


def calc_jacard_sim(label_a, label_b):

    if label_a.shape != label_b.shape:
        raise ValueError('Shapes are not the same')
    if len(label_a.shape) > 1:
        dim=1
    else:
        dim=0
    stacked = torch.stack((label_a, label_b), dim=0)
    upper = torch.min(stacked, dim=0)[0].sum(dim=dim).float()
    lower = torch.max(stacked, dim=0)[0].sum(dim=dim).float()
    epsilon = 1e-8
    value = torch.div(upper, lower+epsilon)
    # value = torch.div(upper, lower)
    return value

class MultiSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, c_treshold=0.3):
        super(MultiSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.c_treshold = c_treshold

    def forward(self, features, labels=None, mask=None, multi=True):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
            multi_val = torch.ones_like(mask).to(device)
        elif labels is not None:
            if len(labels.shape) < 2:
                raise ValueError('This loss only works with multi-label problem')
            labels = labels.contiguous()
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            multi_labels = torch.zeros((batch_size, batch_size), dtype=torch.float32, device=device)
            for x in range(batch_size):
                for y in range(batch_size):
                    multi_labels[x, y] = calc_jacard_sim(labels[x], labels[y])
            mask = torch.where(multi_labels >= self.c_treshold, 1., 0.)

            multi_val = multi_labels

        else:
            mask = mask.float().to(device)
            multi_val = torch.ones_like(mask).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), (self.temperature + 1e-8) )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask and multi_val as per the highlighted concern
        mask = mask.repeat(anchor_count, contrast_count)
        multi_val = multi_val.repeat(anchor_count, contrast_count)

        # Mask-out self-contrast cases correctly as per your instructions
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask


        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        multi_log_prob = log_prob * multi_val

        #print("Contains NaN:", torch.isnan(multi_labels).any().item())

        mean_multi_log_prob_pos = (mask * multi_log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = - (self.temperature / self.base_temperature) * mean_multi_log_prob_pos

        loss = loss.view(anchor_count, batch_size)

        return loss.mean()






import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# LOAD DATASET
df = pd.read_csv('merged_output.csv')


sequences = df['Sequence'].tolist()


for i in range(len(sequences)):
    if len(sequences[i]) > 512:
        sequences[i] = sequences[i][:512]

labels = df[['nucleus', 'exosome', 'cytosol', 'ribosome', 'membrane', 'endoplasmic reticulum']].values


def convert_seqs_to_embeddings(sequences):
    embeddings = []
    for seq in tqdm(sequences, desc="Converting sequences to embeddings"):
        #change T to U
        seq = seq.replace('T', 'U')
        seq = seq.lower()
        #join with space
        seq = ' '.join(seq)
        output_tokens = tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            return_tensors='pt'
        )['input_ids'].to(device)

        with torch.no_grad():
            output_vector = model(output_tokens).last_hidden_state[:, 0, :]
        embeddings.append(output_vector.cpu().numpy().squeeze())

    return embeddings

class RNASeqDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)


embeddings = convert_seqs_to_embeddings(sequences)

embeddings_np = np.array(embeddings)

labels_np = np.array(labels)

#### 5 fold cv

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, average_precision_score, coverage_error, label_ranking_loss, hamming_loss
import torch.optim as optim

# Define the number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to store the metrics across folds
accuracy_scores = []
average_precision_scores = []
coverage_errors = []
ranking_losses = []
hamming_losses = []
one_errors = []

embeddings_np = np.array(embeddings)
labels_np = np.array(labels)

# Parameters and model setup
input_dim = 768  
hidden_dim = 256  
output_dim = 128  
num_heads = 4  
projection_dim = 64

# Loss function
temperature = 0.07
contrast_mode = 'all'
base_temperature = 0.07
c_treshold = 0.3


for fold, (train_index, val_index) in enumerate(kf.split(embeddings_np)):
    print(f"Fold {fold + 1}/{n_splits}")

    # Split the data into train and validation sets for this fold
    train_embeddings, val_embeddings = embeddings_np[train_index], embeddings_np[val_index]
    train_labels, val_labels = labels_np[train_index], labels_np[val_index]

    # Convert to PyTorch datasets
    train_dataset = RNASeqDataset(train_embeddings, train_labels)
    val_dataset = RNASeqDataset(val_embeddings, val_labels)

    # Create DataLoaders for this fold
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model and optimizer
    model = SupConCNNAttention(output_dim=output_dim, projection_dim=projection_dim).to(device)
    criterion = MultiSupConLoss(temperature=temperature, contrast_mode=contrast_mode,
                                base_temperature=base_temperature, c_treshold=c_treshold).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 150
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_dataloader:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            features = model(data).unsqueeze(1)
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader)}')

    # Instantiate the classifier
    linear_classifier = MLDecoder(num_classes=6, decoder_embedding=768, initial_num_features=output_dim).to(device)
    classifier_optimizer = optim.Adam(linear_classifier.parameters(), lr=1e-4)
    criterion_classifier = torch.nn.BCEWithLogitsLoss()

    # Training loop for the classifier
    epochs = 100
    for epoch in range(epochs):
        model.train()
        linear_classifier.train()
        for batch in train_dataloader:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            classifier_optimizer.zero_grad()

            features = model.encoder(data)
            output = linear_classifier(features.detach().unsqueeze(1))
            loss = criterion_classifier(output, labels)
            loss.backward()
            classifier_optimizer.step()

        print(f'Classifier Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # Evaluation on validation set
    all_true_labels = []
    all_pred_probs = []

    model.eval()
    linear_classifier.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            features = model.encoder(data)
            output = linear_classifier(features.unsqueeze(1))
            pred_probs = torch.sigmoid(output)
            all_true_labels.append(labels.cpu().numpy())
            all_pred_probs.append(pred_probs.cpu().numpy())

    all_true_labels = np.vstack(all_true_labels)
    all_pred_probs = np.vstack(all_pred_probs)
    all_pred_labels = (all_pred_probs > 0.5).astype(int)

    # Calculate metrics for this fold
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    average_precision = average_precision_score(all_true_labels, all_pred_probs, average="samples")
    coverage = coverage_error(all_true_labels, all_pred_probs) - 1
    ranking_loss = label_ranking_loss(all_true_labels, all_pred_probs)
    hamming = hamming_loss(all_true_labels, all_pred_labels)
    one_error = np.mean([np.argmax(all_pred_probs[i]) not in np.where(all_true_labels[i] == 1)[0] for i in range(len(all_pred_probs))])

    # Store the metrics
    accuracy_scores.append(accuracy)
    average_precision_scores.append(average_precision)
    coverage_errors.append(coverage)
    ranking_losses.append(ranking_loss)
    hamming_losses.append(hamming)
    one_errors.append(one_error)

# Calculate average metrics across all folds
print(f'Average Accuracy: {np.mean(accuracy_scores)}')
print(f'Average Precision: {np.mean(average_precision_scores)}')
print(f'Average Coverage: {np.mean(coverage_errors)}')
print(f'Average Ranking Loss: {np.mean(ranking_losses)}')
print(f'Average Hamming Loss: {np.mean(hamming_losses)}')
print(f'Average One-error: {np.mean(one_errors)}')
