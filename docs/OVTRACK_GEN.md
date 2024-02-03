# OVTrack Image Generation: 

This document provides a brief overview of the image generation process for OVTrack. The image generation process is used to generate the images from static LVIS images for the OVTrack training. 


## 1. Prepare the Pretrained Models and Annotations.

### 1.1 Prepare the annotations.
Please refer to the instructions in [GET_STARTED.md](GET_STARTED.md) to download the LVIS dataset and prepare the annotations.

### 1.2 Prepare the pretrained models.
Please download the pretrained models from the following links:
https://huggingface.co/CompVis/stable-diffusion-v1-4
When downloaded the model please put it in the `$OVTrack/saved_models` folder.



## 2. Generate the images.

To generate the images, please run the following command:

```shell
tools/run_ovtrack_generation.sh $GPUs $RESULT_FOLDER_NAME
```

Where `$GPUS` is the number of GPUs to use for the generation process and `$RESULT_FOLDER_NAME` is the name of the folder where the generated images will be saved in hdf5 format.
The folder will be created in the `data/tao/ovtrack/` folder.

## 3. Post-processing the generated images.

This step is to merge all the images hdf5 partitions generated in the previous step into a single hdf5 file.

Please use following command to merge the generated hdf5 images into a single file:

```shell
python tools/post_processing_scripts/merge_hdf5.py --h5_folder_name $RESULT_FOLDER_NAME --total_p 8
```

Where `$RESULT_FOLDER_NAME` is the name of the folder where the generated images are saved in hdf5 format and `--total_p` is the number of partitions generated in the previous step.

The merged hdf5 file, and its corresponding annotations (.pkl) will be saved in the `data/tao/ovtrack/` folder.

Next, we need to convert the pkl annotations to json format. Please use the following command to convert the pkl annotations to json format:

```shell
python tools/ post_processing_scripts/generate_json_anno.py --h5_folder_name $RESULT_FOLDER_NAME
```

Please note that we also add original annotations of LVIS into the annotation file, this is because it contains small objects that may not present in the generated images.
Due to the nature of stable diffusion model where the diffusion steps happen in a latent space, it is possible that some small objects maybe missed.
The generated json annotations will be saved in the `data/tao/ovtrack/annotations/` folder.

## 4. Optional: Visualize the generated images.

To visualize the generated images, please follow the instructions in [notebook](../tools/notebook_scripts/visualize_generated_fake_images.ipynb) to run the visualization.
