# Getting Started
This page provides basic tutorials about the usage of OVTrack. For installation instructions, please see [INSTALL.md](INSTALL.md).

## Prepare Datasets

#### Symlink the data

It is recommended to symlink the dataset root to `$OVTrack/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.
Our folder structure follows

#### Download TAO
a. Please follow [TAO download](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md) instructions.

b. Please also prepare the [LVIS dataset](https://www.lvisdataset.org/).

It is recommended to symlink the dataset root to `$OVTrack/data`.

If your folder structure is different, you may need to change the corresponding paths in config files.

Our folder structure follows

```
├── ovtrack
├── tools
├── configs
├── data
    ├── tao
        ├── frames
            ├── train
            ├── val
            ├── test
        ├── annotations
    ├── lvis
        ├── train2017
        ├── annotations    
```

### 2. Install the TETA API

For more details about the installation and usage of the TETA metric, please refer to [TETA](../teta/README.md).



### 3. Generate our annotation files

a. Convert TAO annotation files.
```shell
python tools/convert_datasets/tao2coco.py -t ./data/tao/annotations/
```
b. Generate TAO val v1 file

```shell
python tools/convert_datasets/create_tao_v1.py data/tao/annotations/validation_ours.json
```
If you want to test on the test set split, please download the test set annotation from [here](https://drive.google.com/file/d/1Ug50Nrj0WDyAIpxZxedzz8nRQFMBeYZ7/view?usp=sharing)
We generate TAO test v1 file from the BURST dataset. Thanks the authors for making the annotations available.

c. During the training and inference, we use an additional file which save all class names: [lvis_classes_v1.txt](https://drive.google.com/file/d/1CzyggqLe4aeqmDEEN-aPgIFIqZcR-Dwj/view?usp=sharing).
Please download it and put it in `${LVIS_PATH}/annotations/`.

## Run OVTrack
This codebase is inherited from [mmdetection](https://github.com/open-mmlab/mmdetection).
You can refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md).
You can also refer to the short instructions below.
We provide config files in [configs](../configs).

### Train a model
Coming soon, please stay tuned!
#### CLIP distillation
We follow the ViLD paper to distill the CLIP model. Specifically, we use the implementation from DetPro and also uses its pre-learned prompt. Please refer to [ViLD](https://arxiv.org/abs/2104.13921) and [DetPro](https://github.com/dyabel/detpro) for more details.
You can download the pretrained distillation model from [here](https://drive.google.com/file/d/1XsBIBydGr1uqZQQu7NQ6eGYcmCNpYnDF/view?usp=sharing).

#### Diffusion-based data generation
Coming soon, please stay tuned!
#### Train the track head 
Coming soon, please stay tuned!


### Test a Model with COCO-format

Note that, in this repo, the evaluation metrics are computed with COCO-format.
But to report the results on BDD100K, evaluating with BDD100K-format is required.

- single GPU
- single node multiple GPU
- multiple node

Download the following trained models for testing

- [OVTrack model](https://drive.google.com/file/d/1vDAFRmuNMCwhKtW7KHONpzkooLysU8nX/view?usp=sharing) Please put it in the 'saved_models/ovtrack_detpro_prompt.pth'
- [DetPro prompt](https://drive.google.com/file/d/1N7ht5b44R2LgExhk0smWLydpTO-RuwMe/view?usp=sharing) Please put it in the 'saved_models/pretrained_models/detpro_prompt.pt'

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--cfg-options]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--cfg-options]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `track`.
- `--cfg-options`: If specified, some setting in the used config will be overridden.


#### Test TAO model on Open-vocabulary MOT benchmark
```angular2html
tools/dist_test.sh configs/ovtrack-teta/ovtrack_r50.py saved_models/ovtrack_detpro_prompt.pth 8 25000 --eval track --eval-options resfile_path=results/ovtrack_teta_results/
```

#### Test TAO model on original TAO benchmark

```angular2html
tools/dist_test.sh configs/ovtrack-tao/ovtrack_r50.py saved_models/ovtrack_detpro_prompt.pth 8 25000 --eval track --eval-options resfile_path=results/ovtrack_tao_results/ use_tao_metric=True
```

#### Test on any video with any class prompt
You can download our model trained with the vanilla CLIP text encoder generated prompt from [here](https://drive.google.com/file/d/1FY4xJx3rUpFOx2HS2ntJ5caC2BL5evX8/view?usp=sharing).
To test with custom classes, you need to **modify** the text_input in the [class_name.py](../ovtrack/models/roi_heads/class_name.py) file.
Please note that the visual tracking results depends a lot on **the proper visualization threshold**. The proper threshold value **varys** on the number of testing classes due to the softmax function. 
Thus, you need to properly tune those values to get ideal visual tracking results. The recommanded values to tune are following: `init_score_thr` and `obj_score_thr` in `tracker`, the `score_thr` in the `rcnn_test_cfg`, and visualization threshold in the visualizer.
You can run following command to test on any video.
```angular2html

python tools/inference.py configs/ovtrack-custom/ovtrack_r50.py saved_models/ovtrack/ovtrack_vanilla_clip_prompt.pth --video YOUR_VIDEO.mp4 --out_frame_dir YOUR_OUTPUT_FRAMES_DIR --out_video_dir YOUR_OUTPUT_VIDEOS_DIR --thickness 1
```

   
