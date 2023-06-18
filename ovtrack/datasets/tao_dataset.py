import mmcv
import numpy as np
import os
import pandas as pd
import pickle
import tempfile
import tqdm
from lvis import LVIS, LVISEval, LVISResults
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset
from .parsers import COCO, CocoVID


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


@DATASETS.register_module(force=True)
class TaoDataset(CocoVideoDataset):
    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        if not self.load_as_video:
            data_infos = self.load_lvis_anns(ann_file)
        else:
            data_infos = self.load_tao_anns(ann_file)
        return data_infos

    def load_lvis_anns(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info["filename"] = info["file_name"]
            if info["file_name"].startswith("COCO"):
                # Convert form the COCO 2014 file naming convention of
                # COCO_[train/val/test]2014_000000000000.jpg to the 2017
                # naming convention of 000000000000.jpg
                # (LVIS v1 will fix this naming issue)
                info["filename"] = info["file_name"][-16:]
            else:
                info["filename"] = info["file_name"]
            data_infos.append(info)
        return data_infos

    def load_tao_anns(self, ann_file):
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            if self.key_img_sampler is not None:
                img_ids = self.key_img_sampling(img_ids, **self.key_img_sampler)
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                if info["file_name"].startswith("COCO"):
                    # Convert form the COCO 2014 file naming convention of
                    # COCO_[train/val/test]2014_000000000000.jpg to the 2017
                    # naming convention of 000000000000.jpg
                    # (LVIS v1 will fix this naming issue)
                    info["filename"] = info["file_name"][-16:]
                else:
                    info["filename"] = info["file_name"]
                data_infos.append(info)
        return data_infos

    def _track2json(self, results):
        """Convert tracking results to TAO json style."""

        inds = [i for i, _ in enumerate(self.data_infos) if _["frame_id"] == 0]
        num_vids = len(inds)
        inds.append(len(self.data_infos))
        results = [results[inds[i] : inds[i + 1]] for i in range(num_vids)]
        img_infos = [self.data_infos[inds[i] : inds[i + 1]] for i in range(num_vids)]

        json_results = []
        max_track_id = 0
        print("Start format track json")
        for _img_infos, _results in tqdm.tqdm(zip(img_infos, results)):
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
        print("Start format det json")
        for idx in tqdm.tqdm(range(len(self))):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data["image_id"] = img_id
                    data["bbox"] = self.xyxy2xywh(bboxes[i])
                    data["score"] = float(bboxes[i][-1])
                    # if the object detecor is trained on 1230 classes(lvis 0.5).
                    if len(result) != len(self.cat_ids):
                        data["category_id"] = label + 1
                    else:
                        data["category_id"] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def format_results(self, results, resfile_path=None, tcc=True):
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
        assert "bbox_results" in results

        if resfile_path is None:
            tmp_dir = tempfile.TemporaryDirectory()
            resfile_path = tmp_dir.name
        else:
            tmp_dir = None
        os.makedirs(resfile_path, exist_ok=True)
        result_files = dict()

        bbox_results = self._det2json(results["bbox_results"])
        result_files["bbox"] = f"{resfile_path}/tao_bbox.json"
        mmcv.dump(bbox_results, result_files["bbox"])

        track_results = self._track2json(results["track_results"])
        if tcc:
            track_results = majority_vote(track_results)
        result_files["track"] = f"{resfile_path}/tao_track.json"
        mmcv.dump(track_results, result_files["track"])

        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric=["track"],
        logger=None,
        resfile_path=None,
        use_tao_metric=False,
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

        result_files, tmp_dir = self.format_results(results, resfile_path)

        eval_results = dict()

        if "track" in metrics and use_tao_metric:
            from tao.toolkit.tao import TaoEval

            print_log("Evaluating TAO results...", logger)
            tao_eval = TaoEval(self.ann_file, result_files["track"])
            tao_eval.params.img_ids = self.img_ids
            tao_eval.params.cat_ids = self.cat_ids
            tao_eval.params.iou_thrs = np.array([0.5, 0.75])
            tao_eval.run()

            tao_eval.print_results()
            tao_results = tao_eval.get_results()
            for k, v in tao_results.items():
                if isinstance(k, str) and k.startswith("AP"):
                    key = "track_{}".format(k)
                    val = float("{:.3f}".format(float(v)))
                    eval_results[key] = val

        if "track" in metrics and not use_tao_metric:
            import teta

            # Command line interface:
            default_eval_config = teta.config.get_default_eval_config()
            # print only combined since TrackMAP is undefined for per sequence breakdowns
            default_eval_config["PRINT_ONLY_COMBINED"] = True
            default_eval_config["DISPLAY_LESS_PROGRESS"] = True
            default_eval_config["OUTPUT_TEM_RAW_DATA"] = True
            default_eval_config["NUM_PARALLEL_CORES"] = 8
            default_dataset_config = teta.config.get_default_dataset_config()
            default_dataset_config["TRACKERS_TO_EVAL"] = ["OVTrack"]
            default_dataset_config["GT_FOLDER"] = self.ann_file
            default_dataset_config["OUTPUT_FOLDER"] = resfile_path
            default_dataset_config["TRACKER_SUB_FOLDER"] = os.path.join(
                resfile_path, "tao_track.json"
            )

            evaluator = teta.Evaluator(default_eval_config)
            dataset_list = [teta.datasets.TAO(default_dataset_config)]
            print("Overall classes performance")
            evaluator.evaluate(dataset_list, [teta.metrics.TETA()])

            eval_results_path = os.path.join(
                resfile_path, "OVTrack", "teta_summary_results.pth"
            )
            eval_res = pickle.load(open(eval_results_path, "rb"))

            base_class_synset = set(
                [
                    c["name"]
                    for c in self.coco.dataset["categories"]
                    if c["frequency"] != "r"
                ]
            )
            novel_class_synset = set(
                [
                    c["name"]
                    for c in self.coco.dataset["categories"]
                    if c["frequency"] == "r"
                ]
            )

            compute_teta_on_ovsetup(eval_res, base_class_synset, novel_class_synset)

        if "bbox" in metrics:
            print_log("Evaluating detection results...", logger)
            lvis_gt = LVIS(self.ann_file)
            lvis_dt = LVISResults(lvis_gt, result_files["bbox"])
            lvis_eval = LVISEval(lvis_gt, lvis_dt, "bbox")
            lvis_eval.params.imgIds = self.img_ids
            lvis_eval.params.catIds = self.cat_ids
            lvis_eval.evaluate()
            lvis_eval.accumulate()
            lvis_eval.summarize()
            lvis_eval.print_results()
            lvis_results = lvis_eval.get_results()
            for k, v in lvis_results.items():
                if k.startswith("AP"):
                    key = "{}_{}".format("bbox", k)
                    val = float("{:.3f}".format(float(v)))
                    eval_results[key] = val
            ap_summary = " ".join(
                [
                    "{}:{:.3f}".format(k, float(v))
                    for k, v in lvis_results.items()
                    if k.startswith("AP")
                ]
            )
            eval_results["bbox_mAP_copypaste"] = ap_summary

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results


def compute_teta_on_ovsetup(teta_res, base_class_names, novel_class_names):
    if "COMBINED_SEQ" in teta_res:
        teta_res = teta_res["COMBINED_SEQ"]

    frequent_teta = []
    rare_teta = []
    for key in teta_res:
        if key in base_class_names:
            frequent_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))
        elif key in novel_class_names:
            rare_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))

    print("Base and Novel classes performance")

    # print the header
    print(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "TETA50:",
            "TETA",
            "LocS",
            "AssocS",
            "ClsS",
            "LocRe",
            "LocPr",
            "AssocRe",
            "AssocPr",
            "ClsRe",
            "ClsPr",
        )
    )

    if frequent_teta:
        freq_teta_mean = np.mean(np.stack(frequent_teta), axis=0)

        # print the frequent teta mean
        print("{:<10} ".format("Base"), end="")
        print(*["{:<10.3f}".format(num) for num in freq_teta_mean])

    else:
        print("No Base classes to evaluate!")
        freq_teta_mean = None
    if rare_teta:
        rare_teta_mean = np.mean(np.stack(rare_teta), axis=0)

        # print the rare teta mean
        print("{:<10} ".format("Novel"), end="")
        print(*["{:<10.3f}".format(num) for num in rare_teta_mean])
    else:
        print("No Novel classes to evaluate!")
        rare_teta_mean = None

    return freq_teta_mean, rare_teta_mean
