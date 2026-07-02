import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from skimage.segmentation import slic


class BCE3wminmaxLoss(nn.Module):
    """BCE3 loss plus min/max consistency for AV3 segmentation."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, imgs, pred_vessels, vessels, mask, cl_features):
        mask = torch.round(mask[:, 0, :, :])

        pred_a = pred_vessels[:, 0, :, :]
        pred_v = pred_vessels[:, 1, :, :]
        pred_vt = pred_vessels[:, 2, :, :]

        gt_a = vessels[:, 0, :, :]
        gt_v = vessels[:, 1, :, :]
        gt_vt = vessels[:, 2, :, :]

        change_tarall = torch.where(
            (gt_a == 1) & (gt_v == 0), torch.min(pred_a, pred_vt),
            torch.where(
                (gt_a == 0) & (gt_v == 1), torch.min(pred_v, pred_vt),
                torch.where(
                    (gt_a == 1) & (gt_v == 1),
                    torch.min(pred_a, torch.min(pred_v, pred_vt)),
                    pred_vt,
                ),
            ),
        )

        uncertain = gt_vt - gt_v - gt_a
        uncertain[uncertain < 0] = 0

        mask_unknown = mask - uncertain
        mask_unknown[mask_unknown < 0] = 0

        loss = self.loss(pred_a[mask_unknown > 0.5], gt_a[mask_unknown > 0.5])
        loss += self.loss(pred_v[mask_unknown > 0.5], gt_v[mask_unknown > 0.5])
        loss += self.loss(pred_vt[mask > 0.], gt_vt[mask > 0.])
        loss += self.loss(change_tarall[mask > 0.], gt_vt[mask > 0.])
        return loss

    def save_predicted(self, prediction, fname):
        vutils.save_image(self.process_predicted(prediction), fname)

    def process_predicted(self, prediction):
        return torch.sigmoid(prediction.clone())


class BCE3Loss(nn.Module):
    """BCE loss for artery, vein, and vessel-tree channels."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, imgs, pred_vessels, vessels, mask, cl_features):
        mask = torch.round(mask[:, 0, :, :])

        pred_a = pred_vessels[:, 0, :, :]
        pred_v = pred_vessels[:, 1, :, :]
        pred_vt = pred_vessels[:, 2, :, :]

        gt_a = vessels[:, 0, :, :]
        gt_v = vessels[:, 1, :, :]
        gt_vt = vessels[:, 2, :, :]

        uncertain = gt_vt - gt_v - gt_a
        uncertain[uncertain < 0] = 0

        mask_unknown = mask - uncertain
        mask_unknown[mask_unknown < 0] = 0

        loss = self.loss(pred_a[mask_unknown > 0.5], gt_a[mask_unknown > 0.5])
        loss += self.loss(pred_v[mask_unknown > 0.5], gt_v[mask_unknown > 0.5])
        loss += self.loss(pred_vt[mask > 0.], gt_vt[mask > 0.])
        return loss

    def save_predicted(self, prediction, fname):
        vutils.save_image(self.process_predicted(prediction), fname)

    def process_predicted(self, prediction):
        return torch.sigmoid(prediction.clone())


class RRLoss(nn.Module):
    """Recursive refinement loss.

    base_criterion is applied at every RRWNet prediction. add_criterion, when
    configured, is applied once to the final prediction.
    """

    def __init__(self, base_criterion, add_criterion=None):
        super().__init__()
        self.base_criterion = base_criterion
        self.add_criterion = add_criterion

    def forward(self, imgs, predictions, gt, mask, cl_features, epoch, superpixel_mask=None):
        loss_1 = self.base_criterion(imgs, predictions[0], gt, mask, cl_features[0])
        if len(predictions) == 1:
            return loss_1

        loss_2 = self.base_criterion(imgs, predictions[1], gt, mask, cl_features[1])
        if len(predictions) > 2:
            for i, (prediction, cl_feature) in enumerate(zip(predictions[2:], cl_features[2:]), 2):
                loss_2 += i * self.base_criterion(imgs, prediction, gt, mask, cl_feature)

            k = len(predictions[1:])
            loss_2 *= 1 / (0.5 * k * (k + 1))

        loss = loss_1 + loss_2

        if self.add_criterion:
            add_weight = 0.009
            loss_add = self.add_criterion(
                imgs, predictions[-1], gt, mask, cl_features[-1], superpixel_mask
            )
            loss += loss_add * add_weight

        return loss

    def save_predicted(self, predictions, fname):
        self.base_criterion.save_predicted(predictions[-1], fname)

    def process_predicted(self, predictions):
        return [self.base_criterion.process_predicted(prediction) for prediction in predictions]


class SupConLoss(nn.Module):
    """Supervised contrastive loss used by SpCLLoss."""

    def __init__(self, threshold=0.1, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, contrastive_method='simclr'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.threshold = threshold
        self.contrastive_method = contrastive_method

    def _cosine_simililarity(self, x, y):
        return self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))

    def forward(self, features, labels=None, mask=None, weight=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown contrast_mode: {self.contrast_mode}')

        anchor_dot_contrast = self._cosine_simililarity(anchor_feature, contrast_feature)
        anchor_dot_contrast = torch.div(anchor_dot_contrast, self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        if weight is not None:
            loss = loss * weight
        return loss


def _superpixel_labels(img, superpixel):
    if superpixel is not None:
        return torch.as_tensor(superpixel)
    img_np = img.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
    labels = slic(img_np, n_segments=25, compactness=10.)
    return torch.from_numpy(labels)


def SpConLoss(feature, img, superpixel=None):
    batch_size = feature.shape[0]
    labels = _superpixel_labels(img, superpixel).unsqueeze(0)

    _, _, fmap_h, fmap_w = feature.shape
    labels = F.interpolate(
        labels.unsqueeze(1).float().to(feature.device),
        size=(fmap_h, fmap_w),
        mode='bilinear',
        align_corners=True,
    ).squeeze(1)

    criterion = SupConLoss(temperature=0.1, contrastive_method='gcl').to(feature.device)
    loss_sp_intra = 0
    features = torch.transpose(feature.view(feature.shape[1], -1), 0, 1)
    sp_features = torch.cat([features.unsqueeze(1), features.unsqueeze(1)], dim=1)
    sp_labels = labels.view(-1)

    chunk_count = 10
    chunk_size = features.shape[0] // chunk_count
    if chunk_size == 0:
        return criterion(sp_features, labels=sp_labels) / batch_size

    for i in range(chunk_count):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < chunk_count - 1 else features.shape[0]
        loss_sp_intra += criterion(sp_features[start:end], labels=sp_labels[start:end])

    return loss_sp_intra / chunk_count / batch_size


class SpCLLoss(nn.Module):
    """Superpixel contrastive loss applied to the final RRWNet prediction."""

    def forward(self, imgs, pred_vessels, vessels, mask, cl_features, superpixel=None):
        return SpConLoss(cl_features, imgs, superpixel)

    def save_predicted(self, prediction, fname):
        vutils.save_image(self.process_predicted(prediction), fname)

    def process_predicted(self, prediction):
        return torch.sigmoid(prediction.clone())
