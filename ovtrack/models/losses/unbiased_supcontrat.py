from __future__ import print_function

import torch
import torch.nn as nn
from mmdet.models import LOSSES


@LOSSES.register_module()
class UnbiasedSupConLoss(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        pos_normalize=True,
        loss_weight=1,
    ):
        super(UnbiasedSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.pos_normalize = pos_normalize
        self.loss_weight = loss_weight

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
            valid_mask = mask.sum(1) != 1
            labels = labels[valid_mask]
            features = features[valid_mask]
            mask = torch.eq(labels, labels.T).float().to(device)
            batch_size = features.shape[0]
            if batch_size == 0:
                return torch.tensor([0.0], requires_grad=True)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        if min(anchor_dot_contrast.shape) != 0:
            # return torch.tensor(0.0).to(anchor_dot_contrast.device)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
        else:
            logits = anchor_dot_contrast
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        if self.pos_normalize:
            pos_norm = torch.div(mask, mask.sum(1).reshape(-1, 1))
            exp_logits = (torch.exp(logits) * logits_mask) * pos_norm + (
                torch.exp(logits) * logits_mask
            ) * torch.logical_not(mask)
        else:
            exp_logits = torch.exp(logits) * logits_mask
        exp_logits_input = exp_logits.sum(1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_input)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss * self.loss_weight
