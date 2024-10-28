
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

