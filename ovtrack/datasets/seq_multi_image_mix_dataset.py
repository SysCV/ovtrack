import collections
import copy
import numpy as np
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import DATASETS, PIPELINES


@DATASETS.register_module()
class SeqMultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None. It is deprecated.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self,
                 dataset,
                 pipeline,
                 dynamic_scale=None,
                 skip_type_keys=None,
                 generated_img=False):
        if dynamic_scale is not None:
            raise RuntimeError(
                'dynamic_scale is deprecated. Please use Resize pipeline '
                'to achieve similar functions')
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = build_from_cfg(transform, PIPELINES)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.PALETTE = getattr(dataset, 'PALETTE', None)
        if hasattr(self.dataset, 'flag'):
            self.flag = dataset.flag
        self.num_samples = len(dataset)
        self.generated_img = generated_img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        while True:
            results = copy.deepcopy(self.dataset[idx])
            if self.generated_img:
                if results[0]['filename'] != results[1]['filename']:
                    self.update_skip_type_keys(['SeqMosaic', 'SeqMixUp'])
                else:
                    self._skip_type_keys = None

            for (transform, transform_type) in zip(self.pipeline,
                                                   self.pipeline_types):
                if self._skip_type_keys is not None and \
                        transform_type in self._skip_type_keys:
                    continue

                if hasattr(transform, 'get_indexes'):
                    indexes = transform.get_indexes(self.dataset)
                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]
                    mix_results = [
                        copy.deepcopy(self.dataset[index]) for index in indexes
                    ]
                    for i, _result in enumerate(results):
                        _mix_result = []
                        for item in mix_results:
                            _mix_result.append(item[i])
                        _result['mix_results'] = _mix_result

                    # results['mix_results'] = mix_results

                results = transform(results)


                if 'mix_results' in results:
                    results.pop('mix_results')

            if isinstance(results, dict):
                if len(results['gt_bboxes'].data) == 0 or len(results['ref_gt_bboxes'].data) == 0:
                    idx = self._rand_another(idx)
                    continue
            if isinstance(results, list):
                if len(results[0]['gt_labels'].data) == 0 or len(results[1]['gt_labels'].data) == 0 or len(results[0]['ref_gt_labels'].data) == 0 or len(results[1]['ref_gt_labels'].data) == 0:
                    idx = self._rand_another(idx)
                    continue
            return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)