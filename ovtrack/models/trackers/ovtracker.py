import cv2
import mmcv
import numpy as np
import os
import random
import seaborn as sns
import torch
from collections import defaultdict
from mmcv.image import imread, imwrite
from mmcv.visualization import color_val, imshow
from mmdet.core import bbox_overlaps

from ovtrack.core import cal_similarity
from ..builder import TRACKERS


@TRACKERS.register_module()
class OVTracker(object):
    def __init__(
        self,
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_frames=10,
        momentum_embed=0.8,
        momentum_obj_score=0.5,
        distractor_nms_thr=0.3,
        distractor_score_thr=0.5,
        match_metric="bisoftmax",
        match_with_cosine=True,
        contrastive_thr=0.5,
    ):
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr

        self.memo_frames = memo_frames
        self.momentum_embed = momentum_embed
        self.momentum_obj_score = momentum_obj_score
        self.distractor_nms_thr = distractor_nms_thr
        self.distractor_score_thr = distractor_score_thr
        assert match_metric in ["bisoftmax", "cosine"]
        self.match_metric = match_metric
        self.match_with_cosine = match_with_cosine
        self.contrastive_thr = contrastive_thr

        self.reset()

    def reset(self):
        self.num_tracklets = 0
        self.tracklets = dict()
        # for analysis
        self.pred_tracks = defaultdict(lambda: defaultdict(list))
        self.gt_tracks = defaultdict(lambda: defaultdict(list))

    @property
    def valid_ids(self):
        valid_ids = []
        for k, v in self.gt_tracks.items():
            valid_ids.extend(v["ids"])
        return list(set(valid_ids))

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, labels, embeds, cls_embeds, frame_id):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, cls_embed, label in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            embeds[tracklet_inds],
            cls_embeds[tracklet_inds],
            labels[tracklet_inds],
        ):
            id = int(id)
            if id in self.tracklets:
                self.tracklets[id]["bboxes"].append(bbox)
                self.tracklets[id]["labels"].append(label)
                self.tracklets[id]["embeds"] = (
                    1 - self.momentum_embed
                ) * self.tracklets[id]["embeds"] + self.momentum_embed * embed
                self.tracklets[id]["cls_embeds"] = cls_embed
                self.tracklets[id]["frame_ids"].append(frame_id)
            else:
                self.tracklets[id] = dict(
                    bboxes=[bbox],
                    labels=[label],
                    embeds=embed,
                    cls_embeds=cls_embed,
                    frame_ids=[frame_id],
                )

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v["frame_ids"][-1] >= self.memo_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

    @property
    def memo(self):
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_embeds = []
        memo_cls_embeds = []
        for k, v in self.tracklets.items():
            memo_ids.append(k)
            memo_bboxes.append(v["bboxes"][-1][None, :])
            memo_labels.append(v["labels"][-1].view(1, 1))
            memo_embeds.append(v["embeds"][None, :])
            memo_cls_embeds.append(v["cls_embeds"][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_cls_embeds = torch.cat(memo_cls_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_cls_embeds,
            memo_ids.squeeze(0),
        )

    def init_tracklets(self, ids, obj_scores):
        new_objs = (ids == -1) & (obj_scores > self.init_score_thr).cpu()
        num_new_objs = new_objs.sum()
        ids[new_objs] = torch.arange(
            self.num_tracklets, self.num_tracklets + num_new_objs, dtype=torch.long
        )
        self.num_tracklets += num_new_objs
        return ids

    def match(
        self,
        bboxes,
        labels,
        embeds,
        cls_embeds,
        frame_id,
        temperature=-1,
        method="ovtrack-teta",
        **kwargs
    ):
        """

        Args:
            bboxes:
            labels:
            track_feats: if use transformer method, the track_feats will be the encoder feats
            cls_feats: if use transformer method, the cls_feats will be the decoder feats
            frame_id:
            temperature:
            method: 'TETer'| 'oracle' | 'appearance' | 'contrastive'
            **kwargs:

        Returns:

        """

        if embeds is None:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            return bboxes, labels, ids

        if method == "ovtrack-teta":
            # match if buffer is not empty
            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="inter"
            )
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo

                if self.match_metric == "bisoftmax":
                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )

                    exps = torch.exp(sims)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")

                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")

                    cls_sims = cal_similarity(
                        cls_embeds,
                        memo_cls_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = cls_sims > self.contrastive_thr
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)

                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0

            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        elif method == "ovtrack-tao":

            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="intra"
            )
            # match if buffer is not empty
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo
                if self.match_metric == "bisoftmax":
                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    exps = torch.exp(sims) * cat_same.to(sims.device)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cos_scores *= cat_same.to(cos_scores.device)
                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0
                        m = self.momentum_obj_score
                        bboxes[i, -1] = (
                            m * bboxes[i, -1] + (1 - m) * memo_bboxes[memo_ind, -1]
                        )
            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        else:
            raise NotImplementedError

        return bboxes, labels, ids

    def remove_distractor(
        self,
        bboxes,
        labels,
        track_feats,
        cls_feats,
        object_score_thr=0.5,
        distractor_nms_thr=0.3,
        softmax_feats=None,
        nms="inter",
    ):

        # all objects is valid here
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(
            bboxes[:, -1] < object_score_thr, as_tuple=False
        ).squeeze(1)
        if nms == "inter":
            ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
        elif nms == "intra":
            cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
            ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
            ious *= cat_same.to(ious.device)
        else:
            raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        embeds = track_feats[valid_inds]
        cls_embeds = cls_feats[valid_inds]
        if softmax_feats is not None:
            softmax_feats = softmax_feats[valid_inds]

        return bboxes, labels, embeds, cls_embeds, softmax_feats
