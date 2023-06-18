import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import PIL.Image
import torch
from mmdet.core import bbox2result
from mmdet.models import TwoStageDetector
from PIL import Image

from ovtrack.core import imshow_tracks, restore_result, track2result
from ..builder import MODELS, build_tracker


@MODELS.register_module()
class OVTrack(TwoStageDetector):
    def __init__(
        self,
        tracker=None,
        freeze_detector=False,
        save_dets=False,
        pickle_path=None,
        use_fixed_dets=False,
        method="ovtrack-teta",
        *args,
        **kwargs
    ):
        self.prepare_cfg(kwargs)
        super().__init__(*args, **kwargs)
        self.tracker_cfg = tracker
        self.method = method
        print(self.method)
        self.freeze_detector = freeze_detector
        self.save_dets = save_dets
        self.use_fixed_dets = use_fixed_dets
        self.pickle_path = pickle_path
        if save_dets:
            assert pickle_path is not None, "please indicate pickle path"
            assert (
                use_fixed_dets == False
            ), "use_fixed_dets and save_dets should not be used in the same time"
        if self.freeze_detector:
            self._freeze_detector()

    def _freeze_detector(self):

        self.detector = [
            self.backbone,
            self.neck,
            self.rpn_head,
            self.roi_head.bbox_head,
        ]

        for model in self.detector:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def prepare_cfg(self, kwargs):
        if kwargs.get("train_cfg", False):
            if kwargs["train_cfg"].get("embed", None):
                kwargs["roi_head"]["track_train_cfg"] = kwargs["train_cfg"].get(
                    "embed", None
                )

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
            return x

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_match_indices,
        ref_img,
        ref_img_metas,
        ref_gt_bboxes,
        ref_gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        ref_gt_bboxes_ignore=None,
        **kwargs
    ):

        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_img)

        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)

        if self.freeze_detector:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
            losses.update(rpn_losses)

        ref_proposals = self.rpn_head.simple_test_rpn(ref_x, ref_img_metas)

        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_match_indices,
            ref_x,
            ref_img_metas,
            ref_proposals,
            ref_gt_bboxes,
            ref_gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            ref_gt_bboxes_ignore,
            **kwargs
        )
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False):

        assert self.roi_head.with_track, "Track head must be implemented."
        img_name = img_metas[0]["filename"]
        if img_name is not None:
            pickle_name = img_name.replace("/", "-").replace(".jpg", ".pth")

        frame_id = img_metas[0].get("frame_id", -1)
        if frame_id == 0:
            self.init_tracker()
        if self.roi_head.custom_classes:
            # The threshold needs to be adjusted according to the number of test classes in the custom case
            self.tracker.init_score_thr = 1 / (self.roi_head.num_classes + 1) + 0.1
            self.tracker.obj_score_thr = 1 / (self.roi_head.num_classes + 1) + 0.05

        x = self.extract_feat(img)

        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        outputs = self.roi_head.simple_test(x, img_metas, proposal_list, rescale)

        if len(outputs) == 4:
            det_bboxes, det_labels, cem_feats, track_feats = outputs
            if cem_feats is None:
                cem_feats = copy.deepcopy(track_feats)
        elif len(outputs) == 3:
            det_bboxes, det_labels, track_feats = outputs
            cem_feats = copy.deepcopy(track_feats)

        if track_feats is not None:

            bboxes, labels, ids = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                embeds=track_feats,
                cls_embeds=cem_feats,
                frame_id=frame_id,
                method=self.method,
            )

        bbox_result = bbox2result(det_bboxes, det_labels, self.roi_head.num_classes)

        if track_feats is not None:
            track_result = track2result(bboxes, labels, ids, self.roi_head.num_classes)
        else:
            track_result = [
                np.zeros((0, 6), dtype=np.float32)
                for i in range(self.roi_head.num_classes)
            ]
        results = dict(bbox_results=bbox_result, track_results=track_result)

        return results

    def show_result(
        self,
        img,
        result,
        vis_thr=0.2,
        thickness=2,
        font_scale=0.1,
        show=False,
        out_file=None,
        wait_time=0,
        backend="cv2",
        **kwargs
    ):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                The value of key 'track_results' is ndarray with shape (n, 6)
                in [id, tl_x, tl_y, br_x, br_y, score] format.
                The value of key 'bbox_results' is ndarray with shape (n, 5)
                in [tl_x, tl_y, br_x, br_y, score] format.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_result = result.get("track_results", None)
        bboxes, labels, ids = restore_result(track_result, return_ids=True)
        valid_ids = bboxes[:, -1] > vis_thr
        bboxes = bboxes[valid_ids]
        labels = labels[valid_ids]
        ids = ids[valid_ids]

        img = imshow_tracks(
            img,
            bboxes,
            labels,
            ids,
            classes=self.roi_head.CLASSES,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            out_file=out_file,
            wait_time=wait_time,
            backend=backend,
        )
        return img
