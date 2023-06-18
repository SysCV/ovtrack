import json
import mmcv
import os
import pandas as pd
import tempfile
import tqdm
from bdd100k.common.utils import load_bdd100k_config
from bdd100k.label.to_scalabel import bdd100k_to_scalabel
from mmdet.datasets import DATASETS
from scalabel.eval.mot import acc_single_video_mot, evaluate_track
from scalabel.label.io import group_and_sort, load

from .coco_video_dataset import CocoVideoDataset


def majority_vote(prediction):

    tid_res_mapping = {}
    for res in prediction:
        tid = res["track_id"]
        if tid not in tid_res_mapping:
            tid_res_mapping[tid] = [res]
        else:
            tid_res_mapping[tid].append(res)
    # change the results to data frame
    df_pred_res = pd.DataFrame(prediction)
    # group the results by track_id
    groued_df_pred_res = df_pred_res.groupby("track_id")

    # change the majority
    class_by_majority_count_res = []
    for tid, group in tqdm.tqdm(groued_df_pred_res):
        cid = group["category_id"].mode()[0]
        group["category_id"] = cid
        dict_list = group.to_dict("records")
        class_by_majority_count_res += dict_list
    return class_by_majority_count_res


def convert_pred_to_label_format(coco_pred, bdd_cid_cinfo_mapping):
    """
    convert the single prediction result to label format for bdd

    coco_pred:
        'image_id': 1,
         'bbox': [998.872802734375,
          379.5665283203125,
          35.427490234375,
          59.21759033203125],
         'score': 0.9133418202400208,
         'category_id': 1,
         'video_id': 1,
         'track_id': 16

    - labels [ ]: list of dicts
        - id: string
        - category: string
        - box2d:
           - x1: float
           - y1: float
           - x2: float
           - y2: float
    Args:
        coco_pred: coco_pred dict.
        bdd_cid_cinfo_mapping: bdd category id to category infomation mapping.
    Return:
        a new dict in bdd format.
    """
    new_label = {}
    new_label["id"] = coco_pred["track_id"]
    new_label["score"] = coco_pred["score"]
    new_label["category"] = bdd_cid_cinfo_mapping[coco_pred["category_id"]]["name"]
    new_label["box2d"] = {
        "x1": coco_pred["bbox"][0],
        "y1": coco_pred["bbox"][1],
        "x2": coco_pred["bbox"][0] + coco_pred["bbox"][2],
        "y2": coco_pred["bbox"][1] + coco_pred["bbox"][3],
    }
    return new_label


def convert_coco_result_to_bdd(
    new_pred, bdd_cid_cinfo_mapping, imid_iminfo_mapping, vid_vinfo_mapping
):
    """
    Args:
        new_pred: list of coco predictions
        bdd_cid_cinfo_mapping: bdd category id to category infomation mapping.
    Return:
        submitable result for bdd eval
    """

    imid_new_dict_mapping = {}
    for item in tqdm.tqdm(new_pred):
        imid = item["image_id"]
        if imid not in imid_new_dict_mapping:
            new_dict = {}
            new_dict["name"] = imid_iminfo_mapping[imid]["file_name"]
            new_dict["videoName"] = vid_vinfo_mapping[
                imid_iminfo_mapping[imid]["video_id"]
            ]["name"]
            new_dict["frameIndex"] = imid_iminfo_mapping[imid]["frame_id"]
            new_dict["labels"] = [
                convert_pred_to_label_format(item, bdd_cid_cinfo_mapping)
            ]
            imid_new_dict_mapping[imid] = new_dict
        else:
            imid_new_dict_mapping[imid]["labels"].append(
                convert_pred_to_label_format(item, bdd_cid_cinfo_mapping)
            )
    for key in imid_iminfo_mapping:
        if key not in imid_new_dict_mapping:
            new_dict = {}
            new_dict["name"] = imid_iminfo_mapping[key]["file_name"]
            new_dict["videoName"] = vid_vinfo_mapping[
                imid_iminfo_mapping[key]["video_id"]
            ]["name"]
            new_dict["frameIndex"] = imid_iminfo_mapping[key]["frame_id"]
            new_dict["labels"] = []
            imid_new_dict_mapping[key] = new_dict
    return list(imid_new_dict_mapping.values())


@DATASETS.register_module()
class BDDVideoDataset(CocoVideoDataset):

    CLASSES = (
        "pedestrian",
        "rider",
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
        "train",
    )

    def __init__(self, scalabel_gt=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalabel_gt = scalabel_gt

    def _track2json(self, results):
        """Convert tracking results to TAO json style."""
        inds = [i for i, _ in enumerate(self.data_infos) if _["frame_id"] == 0]
        num_vids = len(inds)
        inds.append(len(self.data_infos))
        results = [results[inds[i] : inds[i + 1]] for i in range(num_vids)]
        img_infos = [self.data_infos[inds[i] : inds[i + 1]] for i in range(num_vids)]

        json_results = []
        max_track_id = 0
        for _img_infos, _results in zip(img_infos, results):
            track_ids = []
            for img_info, result in zip(_img_infos, _results):
                img_id = img_info["id"]
                for label in range(len(result)):
                    bboxes = result[label]
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data["image_id"] = img_id
                        data["bbox"] = self.xyxy2xywh(bboxes[i, 1:])
                        data["score"] = float(bboxes[i][-1])
                        if len(result) != len(self.cat_ids):
                            data["category_id"] = label + 1
                        else:
                            data["category_id"] = self.cat_ids[label]
                        data["video_id"] = img_info["video_id"]
                        data["track_id"] = max_track_id + int(bboxes[i][0])
                        track_ids.append(int(bboxes[i][0]))
                        json_results.append(data)
            track_ids = list(set(track_ids))
            if track_ids:
                max_track_id += max(track_ids) + 1

        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data["image_id"] = img_id
                    data["bbox"] = self.xyxy2xywh(bboxes[i])
                    data["score"] = float(bboxes[i][4])
                    # if the object detecor is trained on 1230 classes(lvis 0.5).
                    if len(result) != len(self.cat_ids):
                        data["category_id"] = label + 1
                    else:
                        data["category_id"] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def format_results(
        self, results, resfile_path=None, scalabel=True, tcc=True, metric=["track"]
    ):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, dict), "results must be a list"
        assert "track_results" in results

        if resfile_path is None:
            tmp_dir = tempfile.TemporaryDirectory()
            resfile_path = tmp_dir.name
        else:
            tmp_dir = None
        os.makedirs(resfile_path, exist_ok=True)
        result_files = dict()

        if scalabel:
            bdd_scalabel_gt = json.load(open(self.ann_file))
            bdd_cid_cinfo_mapping = {}
            for c in bdd_scalabel_gt["categories"]:
                if c["id"] not in bdd_cid_cinfo_mapping:
                    bdd_cid_cinfo_mapping[c["id"]] = c
            # imid info mapping
            imid_iminfo_mapping = {}
            for i in bdd_scalabel_gt["images"]:
                if i["id"] not in imid_iminfo_mapping:
                    imid_iminfo_mapping[i["id"]] = i
            # vidid info mapping
            vid_vinfo_mapping = {}
            for i in bdd_scalabel_gt["videos"]:
                if i["id"] not in vid_vinfo_mapping:
                    vid_vinfo_mapping[i["id"]] = i

        if "track_results" in results:
            track_results = self._track2json(results["track_results"])

            if tcc and track_results:
                mc_res = majority_vote(track_results)
            else:
                mc_res = track_results

            if scalabel:
                mc_res_scalabel = convert_coco_result_to_bdd(
                    mc_res,
                    bdd_cid_cinfo_mapping,
                    imid_iminfo_mapping,
                    vid_vinfo_mapping,
                )
                result_files[
                    "track_scalabel"
                ] = f"{resfile_path}/bdd_track_scalabel.json"
                mmcv.dump(mc_res_scalabel, result_files["track_scalabel"])

            result_files["track_coco"] = f"{resfile_path}/bdd_track_coco.json"
            mmcv.dump(mc_res, result_files["track_coco"])

        if "bbox_results" in results:
            bbox_results = self._det2json(results["bbox_results"])
            result_files["bbox"] = f"{resfile_path}/bdd_bbox.json"
            mmcv.dump(bbox_results, result_files["bbox"])

        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric=["bbox", "track"],
        logger=None,
        resfile_path=None,
        bbox_kwargs=dict(
            classwise=False,
            proposal_nums=(100, 300, 1000),
            iou_thrs=None,
            metric_items=None,
        ),
        track_kwargs=dict(
            iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False, nproc=1
        ),
    ):

        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError("metric must be a list or a str.")
        allowed_metrics = ["bbox", "track"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported.")

        if resfile_path is not None:
            if resfile_path.endswith(".json"):
                result_files = json.load(open(resfile_path))
                tmp_dir = None
            else:
                result_files, tmp_dir = self.format_results(
                    results, resfile_path, scalabel=True, tcc=True, metric=metrics
                )
        else:
            result_files, tmp_dir = self.format_results(
                results, resfile_path, scalabel=True, tcc=True, metric=metrics
            )

        if "track" in metrics:

            import teta

            # Command line interface:
            default_eval_config = teta.config.get_default_eval_config()
            default_eval_config["PRINT_ONLY_COMBINED"] = True
            default_eval_config["DISPLAY_LESS_PROGRESS"] = True
            default_eval_config["NUM_PARALLEL_CORES"] = 8
            default_dataset_config = teta.config.get_default_dataset_config()
            default_dataset_config["TRACKERS_TO_EVAL"] = ["TETer"]
            default_dataset_config["GT_FOLDER"] = self.ann_file
            default_dataset_config["TRACKER_SUB_FOLDER"] = result_files["track_coco"]

            evaluator = teta.Evaluator(default_eval_config)
            dataset_list = [teta.datasets.COCO(default_dataset_config)]
            evaluator.evaluate(dataset_list, [teta.metrics.TETA(exhaustive=True)])

            bdd100k_config = load_bdd100k_config(
                "ovtrack/core/evaluation/box_track.toml"
            )

            eval_results = evaluate_track(
                acc_single_video_mot,
                gts=group_and_sort(
                    bdd100k_to_scalabel(
                        load(
                            self.scalabel_gt,
                            min(4, os.cpu_count() if os.cpu_count() else 1),
                        ).frames,
                        bdd100k_config,
                    )
                ),
                results=group_and_sort(
                    bdd100k_to_scalabel(
                        load(
                            result_files["track_scalabel"],
                            min(4, os.cpu_count() if os.cpu_count() else 1),
                        ).frames,
                        bdd100k_config,
                    )
                ),
                config=bdd100k_config.scalabel,
                iou_thr=0.5,
                ignore_iof_thr=0.5,
                nproc=min(4, os.cpu_count() if os.cpu_count() else 1),
            )
        print(eval_results)

        return eval_results.dict()
