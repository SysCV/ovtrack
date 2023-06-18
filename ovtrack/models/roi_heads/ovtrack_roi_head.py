import clip
import os.path as osp
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead
from tqdm import tqdm

from .class_name import *


@HEADS.register_module()
class OVTrackRoIHead(StandardRoIHead):
    def __init__(
        self,
        track_roi_extractor=None,
        track_head=None,
        track_train_cfg=None,
        cem_roi_extractor=None,
        cem_train_cfg=None,
        cem_head=None,
        finetune_track=False,
        kd_weight=256,
        fixed_lambda=None,
        prompt_path=None,
        fix_bg=False,
        ensemble=True,
        custom_classes=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if track_head is not None:
            self.init_track_head(track_roi_extractor, track_head)

        if track_train_cfg is not None:
            self.track_train_cfg = track_train_cfg
            self.init_track_assigner_sampler()

        if cem_head is not None:
            self.init_cem_head(cem_roi_extractor, cem_head)
        else:
            self.cem_head = None

        if cem_train_cfg is not None:
            self.cem_train_cfg = cem_train_cfg
            self.init_cem_assigner_sampler()
        self.finetune_track = finetune_track

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kd_weight = kd_weight
        self.fixed_lambda = fixed_lambda
        self.fix_bg = fix_bg
        self.custom_classes = custom_classes
        if custom_classes:
            self.fixed_lambda = 0.3

        print("fixed_lambda", fixed_lambda)
        print("prompt path", prompt_path)
        self.text_features_for_classes = []
        self.ensemble = ensemble
        print("ensemble:{}".format(self.ensemble))

        if custom_classes:
            self.CLASSES = text_input
            self.num_classes = len(text_input)
        else:
            self.num_classes = self.bbox_head.num_classes
            if self.num_classes == 8:
                self.CLASSES = BDD_CLASSES
            elif self.num_classes == 1203:
                self.CLASSES = LVIS_CLASSES
            else:
                print("For custom classes, please set custom_classes=True")

        if prompt_path is not None:
            save_path = prompt_path
        else:
            save_path = ""

        print("load:", save_path)
        if osp.exists(save_path) and not custom_classes:
            if not self.fix_bg:
                self.text_features_for_classes = torch.load(save_path).squeeze()[
                    : self.bbox_head.num_classes
                ]
                self.text_features_for_classes = self.text_features_for_classes.to(
                    device
                )

            else:
                self.text_features_for_classes = (
                    torch.load(save_path).to(device).squeeze()
                )
                print(self.text_features_for_classes.shape)
        else:
            print("Custom target classes: ", self.CLASSES)
            clip_model, self.preprocess = clip.load("ViT-B/32", device)
            clip_model.eval()
            for child in clip_model.children():
                for param in child.parameters():
                    param.requires_grad = False

            for template in tqdm(template_list):
                print(template)
                text_features_for_classes = torch.cat(
                    [
                        clip_model.encode_text(
                            clip.tokenize(template.format(c)).to(device)
                        ).detach()
                        for c in self.CLASSES
                    ]
                )
                self.text_features_for_classes.append(
                    F.normalize(text_features_for_classes, dim=-1)
                )

            self.text_features_for_classes = torch.stack(
                self.text_features_for_classes
            ).mean(dim=0)

        self.text_features_for_classes = self.text_features_for_classes.float()
        self.text_features_for_classes = F.normalize(
            self.text_features_for_classes, dim=-1
        )

        print(self.text_features_for_classes.shape)

        if not self.fix_bg:
            self.bg_embedding = nn.Linear(1, 512)
            nn.init.xavier_uniform_(self.bg_embedding.weight)
            nn.init.constant_(self.bg_embedding.bias, 0)
        self.projection = nn.Linear(1024, 512)
        # if self.ensemble:
        self.projection_for_image = nn.Linear(1024, 512)
        nn.init.xavier_uniform_(self.projection_for_image.weight)
        nn.init.constant_(self.projection_for_image.bias, 0)

        if self.num_classes == 1230:
            self.base_label_ids = torch.tensor(lvis05_base, device=device)
            self.novel_label_ids = torch.tensor(lvis05_novel, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

        elif self.num_classes == 1203:

            self.base_label_ids = torch.tensor(lvis_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(lvis_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

        elif self.num_classes == 8:
            self.base_label_ids = torch.tensor(bdd_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(bdd_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get("assigner", None):
            self.track_roi_assigner = build_assigner(self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get("sampler", None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self
            )
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    def init_cem_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.cem_train_cfg.get("assigner", None):
            self.cem_roi_assigner = build_assigner(self.cem_train_cfg.assigner)
            self.cem_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.cem_share_assigner = True

        if self.cem_train_cfg.get("sampler", None):
            self.cem_roi_sampler = build_sampler(
                self.cem_train_cfg.sampler, context=self
            )
            self.cem_share_sampler = False
        else:
            self.cem_roi_sampler = self.bbox_sampler
            self.cem_share_sampler = True

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "track_head") and self.track_head is not None

    @property
    def with_cem(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "cem_head") and self.cem_head is not None

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)

    def init_cem_head(self, cem_roi_extractor, cem_head):
        """Initialize ``track_head``"""
        if cem_roi_extractor is not None:
            self.cem_roi_extractor = build_roi_extractor(cem_roi_extractor)
            self.cem_share_extractor = False
        else:
            self.cem_share_extractor = True
            self.cem_roi_extractor = self.bbox_roi_extractor
        self.cem_head = build_head(cem_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        if self.with_track:
            self.track_head.init_weights()
            if not self.track_share_extractor:
                self.track_roi_extractor.init_weights()
        if self.with_cem:
            self.cem_head.init_weights()
            if not self.cem_share_extractor:
                self.cem_roi_extractor.init_weights()

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_match_indices,
        ref_x,
        ref_proposals,
        ref_gt_bboxes,
        ref_gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        ref_gt_bboxes_ignore=None,
        *args,
        **kwargs
    ):
        if not self.finetune_track:
            losses = super().forward_train(
                x,
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
                gt_masks,
                *args,
                **kwargs
            )
        else:
            losses = {}

        num_imgs = len(img_metas)

        if self.with_track or self.with_cem:

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            key_sampling_results, ref_sampling_results = [], []
            for i in range(num_imgs):
                assign_result = self.track_roi_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.track_roi_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                key_sampling_results.append(sampling_result)

                ref_assign_result = self.track_roi_assigner.assign(
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_bboxes_ignore[i],
                    ref_gt_labels[i],
                )
                ref_sampling_result = self.track_roi_sampler.sample(
                    ref_assign_result,
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in ref_x],
                )
                ref_sampling_results.append(ref_sampling_result)

            key_bboxes = [res.pos_bboxes for res in key_sampling_results]

            if self.with_track:
                key_feats = self._track_forward(x, key_bboxes)
                ref_bboxes = [res.bboxes for res in ref_sampling_results]
                ref_feats = self._track_forward(ref_x, ref_bboxes)

                match_feats = self.track_head.match(
                    key_feats, ref_feats, key_sampling_results, ref_sampling_results
                )
                asso_targets = self.track_head.get_track_targets(
                    gt_match_indices, key_sampling_results, ref_sampling_results
                )
                loss_track = self.track_head.loss(*match_feats, *asso_targets)

                losses.update(loss_track)

        return losses

    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[: self.track_roi_extractor.num_inputs], rois
        )
        track_feats = self.track_head(track_feats)
        return track_feats

    def _cem_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        cem_feats = self.cem_roi_extractor(x[: self.cem_roi_extractor.num_inputs], rois)
        cem_feats = self.cem_head(cem_feats)

        return cem_feats

    def _clip_cem_forward(self, x, bboxes):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = bbox2roi(bboxes)
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return region_embeddings

    def simple_test(self, x, img_metas, proposal_list, rescale, **kwargs):

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )

        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]
        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]["scale_factor"]
        ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        return det_bboxes, det_labels, cem_feats, track_feats

    def simple_test_with_fixed_dets(
        self, x, det_bboxes, det_labels, img_metas, **kwargs
    ):

        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]["scale_factor"]
        ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        return det_bboxes, det_labels, cem_feats, track_feats

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        bbox_pred = self.bbox_head(region_embeddings)
        bbox_results = dict(bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, region_embeddings

    def _bbox_forward_for_image(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return None, region_embeddings

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # st1 = time.time()
        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        rois = bbox2roi(proposals)

        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(
                bg_class_embedding, p=2, dim=1
            )
            text_features = torch.cat(
                [self.text_features_for_classes, bg_class_embedding], dim=0
            )
        else:
            text_features = self.text_features_for_classes

        cls_score_text = region_embeddings @ text_features.T
        cls_score_text = cls_score_text / 0.007
        cls_score_text = cls_score_text.softmax(dim=1)

        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(
                region_embeddings_image, p=2, dim=1
            )
            cls_score_image = region_embeddings_image @ text_features.T
            cls_score_image = cls_score_image / 0.007
            cls_score_image[:, -1] = -1e11
            cls_score_image = cls_score_image.softmax(dim=1)

        a = 1 / 3
        if self.ensemble:
            if self.fixed_lambda is not None:
                cls_score = (
                    cls_score_image ** (1 - self.fixed_lambda)
                    * cls_score_text ** self.fixed_lambda
                )
            else:
                cls_score = torch.where(
                    self.novel_index,
                    cls_score_image ** (1 - a) * cls_score_text ** a,
                    cls_score_text ** (1 - a) * cls_score_image ** a,
                )

        else:
            cls_score = cls_score_text

        bbox_pred = bbox_results["bbox_pred"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img
                )
        else:
            bbox_pred = (None,) * len(proposals)

        rcnn_test_cfg.score_thr = 1 / len(text_features) + 0.01

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg,
            )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
