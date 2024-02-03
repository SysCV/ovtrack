import albumentations as A
import argparse
import copy
import cv2
import h5py
import io
import json
import math
import numpy as np
import os
import pickle
import random
import torch
from mmdet.datasets.pipelines.transforms import Pad, Resize
from PIL import Image
from pycocotools.coco import COCO
from torch.cuda.amp import autocast
from .pipelines.stable_diffusion.pipeline_ovtrack_image_generation import StableDiffusionOVTrackPipeline

import random
import torch

# def set_random_seed(seed_value):
#     """
#     Set the seed for various modules to ensure reproducibility.
#     :param seed_value: An integer value for the seed.
#     """
#     random.seed(seed_value)  # Python's built-in random module
#     np.random.seed(seed_value)  # NumPy
#     torch.manual_seed(seed_value)  # PyTorch
#
#     # If you are using CUDA with PyTorch
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
# # Set a fixed seed for reproducibility
# SEED = 42
# set_random_seed(SEED)
# ##################################

device = "cuda"

# replace with your own path
# model_id_or_path  = "/cluster/work/cvl/lisiyu/stable-diffusion-v1-4/"
# model_id_or_path  = "/scratch/snx3000/rxiang/dl/work/stable-diffusion-v1-4"
model_id_or_path  = "./saved_models/stable-diffusion-v1-4/"

# Refactored Transformation Matrices
def get_rotation_matrix(rotate_degrees):
    radian = math.radians(rotate_degrees)
    return np.array([[np.cos(radian), -np.sin(radian), 0.],
                     [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
                    dtype=np.float32)

def get_scaling_matrix(scale_ratio):
    return np.array([[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
                    dtype=np.float32)

def get_shear_matrix(x_shear_degrees, y_shear_degrees):
    x_radian, y_radian = math.radians(x_shear_degrees), math.radians(y_shear_degrees)
    return np.array([[1, np.tan(x_radian), 0.], [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                    dtype=np.float32)

def get_translation_matrix(x, y):
    return np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]], dtype=np.float32)



def generate_random_affine_matrix(
        width, height,
        max_rotate_degree=10.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(0.5, 1.5),
        max_shear_degree=2.0,
):
    # Rotation
    rotation_degree = random.uniform(-max_rotate_degree,
                                     max_rotate_degree)
    rotation_matrix = get_rotation_matrix(rotation_degree)

    # Scaling
    scaling_ratio = random.uniform(scaling_ratio_range[0],
                                   scaling_ratio_range[1])
    scaling_matrix = get_scaling_matrix(scaling_ratio)

    # Shear
    x_degree = random.uniform(-max_shear_degree,
                              max_shear_degree)
    y_degree = random.uniform(-max_shear_degree,
                              max_shear_degree)
    shear_matrix = get_shear_matrix(x_degree, y_degree)

    # Translation
    trans_x = random.uniform(-max_translate_ratio,
                             max_translate_ratio) * width
    trans_y = random.uniform(-max_translate_ratio,
                             max_translate_ratio) * height
    translate_matrix = get_translation_matrix(trans_x, trans_y)

    warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)

    return warp_matrix


# Additional Helper Functions
def compute_ioa(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    area1 = mask1.sum()
    return intersection / area1

def update_annotations(annos, anno_mask_list):
    new_annos = [update_single_annotation(ann, mask) for ann, mask in zip(annos, anno_mask_list)]
    return new_annos

def update_single_annotation(anno, mask):
    new_anno = copy.deepcopy(anno)
    bbox = extract_bboxes(mask[:, :, np.newaxis])[0]
    new_anno['bbox'] = list(bbox)
    new_anno['area'] = mask.sum()
    return new_anno


def merge_masks(mask_list):
    return np.logical_or.reduce(mask_list) if mask_list else []

def extract_bboxes(mask):
    """Compute bounding boxes from masks.

    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (x1, y1, h, w)].

    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2-x1 , y2-y1 ])
    return boxes.astype(np.int32)


# Helper Functions
def filter_annotations(coco_obj, image_id, area_range):
    """
    Filter annotations for a given image based on area range.
    :param coco_obj: COCO dataset object.
    :param image_id: ID of the image.
    :param area_range: Tuple of min and max area.
    :return: Filtered annotations.
    """
    ann_ids = coco_obj.getAnnIds(imgIds=[image_id], areaRng=area_range, iscrowd=None)
    return coco_obj.loadAnns(ann_ids)

def remove_crowd_annotations(annotations):
    """
    Remove annotations labeled as crowd.
    :param annotations: List of annotations.
    :return: Annotations with 'iscrowd' flag not set.
    """
    return [anno for anno in annotations if anno.get('iscrowd', 0) == 0]

def create_transformer(affine_scale, affine_rotate, affine_translate, affine_shear, perspective_scale, flip_probability, brightness_contrast_probability):
    """
    Create a transformer for image augmentation.
    :return: Albumentations Compose object.
    """
    return A.Compose([
        A.Affine(scale=affine_scale, rotate=affine_rotate, translate_percent=affine_translate, shear=affine_shear, p=0.5, mode=0),
        A.Perspective(scale=perspective_scale, keep_size=True, p=0.5),
        A.HorizontalFlip(p=flip_probability),
        A.RandomBrightnessContrast(p=brightness_contrast_probability),
    ])

def apply_transformations(transformer, image, masks):
    """
    Apply transformations to the image and masks.
    :param transformer: Albumentations transformer.
    :param image: Image to transform.
    :param masks: Masks to transform.
    :return: Transformed image and masks.
    """
    transformed = transformer(image=image, masks=masks)
    return transformed['image'], transformed['masks']

def merge_masks(mask_list):
    """
    Merge a list of masks into a single mask.
    :param mask_list: List of masks.
    :return: Merged mask.
    """
    if mask_list:
        return np.logical_or.reduce(mask_list)
    return np.zeros(mask_list[0].shape, dtype=np.uint8)

# Main Function
def get_ovtrack_data_generation_input(
        iminfo, pil_I, coco_obj,
        area_range_min=64 ** 2, area_range_max=32 ** 5,
        affine_scale=(0.6, 1.5), affine_rotate=(-30, 30),
        affine_translate=(0, 0.2), affine_shear=(-15, 15),
        perspective_scale=(0.05, 0.2), flip_probability=0.5,
        brightness_contrast_probability=0.2):
    """
    Generate transformed image and mask data for object tracking.
    :param iminfo: Image information.
    :param pil_I: PIL image.
    :param coco_obj: COCO object instance.
    :param area_range_min: Minimum area for annotation inclusion.
    :param area_range_max: Maximum area for annotation inclusion.
    :param affine_scale: Scale range for affine transformation.
    :param affine_rotate: Rotation range for affine transformation.
    :param affine_translate: Translation range for affine transformation.
    :param affine_shear: Shear range for affine transformation.
    :param perspective_scale: Scale for perspective transformation.
    :param flip_probability: Probability for horizontal flip.
    :param brightness_contrast_probability: Probability for random brightness/contrast.
    :return: Tuple of transformed PIL images, crop size, initial size, and updated annotation list.
    """
    imid = iminfo['id']
    I = np.array(pil_I)

    anns = filter_annotations(coco_obj, imid, (area_range_min, area_range_max))
    anns = remove_crowd_annotations(anns)

    if not anns:
        return None, None, None, None, None

    # Sort annotations by area in descending order
    sorted_anns = sorted(anns, key=lambda item: item['area'], reverse=True)

    # Generate masks for annotations
    mask_dict = {ann['id']: coco_obj.annToMask(ann) for ann in sorted_anns}
    mask_list = list(mask_dict.values())

    # Define random geometric transform with color contrast
    transformer = create_transformer(affine_scale,
                                     affine_rotate,
                                     affine_translate,
                                     affine_shear,
                                     perspective_scale,
                                     flip_probability,
                                     brightness_contrast_probability)
    # Apply transformations
    transformed_image, transformed_masks = apply_transformations(transformer, I, mask_list)

    # Update annotations
    updated_ann_list = update_annotations(sorted_anns, transformed_masks)

    # Merge original and transformed images
    out_mask = (transformed_image == 0)
    new_img = (I * out_mask + transformed_image * (1 - out_mask)).astype(np.uint8)

    # Process mask
    mask_f = merge_masks(transformed_masks).astype(np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    new_mask = cv2.dilate(mask_f, kernel, iterations=2)

    # Resize and pad image and mask
    resize_img_with_ratio = Resize(img_scale=[(768, 512)], keep_ratio=True)
    pad_32 = Pad(size_divisor=64)
    init_mask = Image.fromarray(new_mask * 255)
    init_img = Image.fromarray(new_img)
    init_size = init_img.size

    results = {'img': np.array(init_img)}
    mask_results = {'img': np.array(init_mask)}
    results = resize_img_with_ratio(results)
    results = pad_32(results)
    mask_results = resize_img_with_ratio(mask_results)
    mask_results = pad_32(mask_results)

    init_img = Image.fromarray(results['img'])
    mask_img = Image.fromarray(mask_results['img'])
    crop_size = tuple(np.rint(results['scale_factor'][:2] * init_size))

    return init_img, mask_img, crop_size, init_size, updated_ann_list


def process_image(imid, args, coco_obj, ovtrack_gen_pipe, out_dataset, imid_iminfo_mapping, imid_caption_mapping, img_client, final_anno_list):
    img_info = imid_iminfo_mapping[imid]
    img_name = img_info['file_name']

    prompt, guidance_scale = get_caption_and_guidance_scale(args.cap_json, imid, imid_caption_mapping)

    if args.h5_img:
        value_buf = img_client[img_name]
        out_dataset.create_dataset(f"{img_name}", data=value_buf)
        image_data = np.array(value_buf)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    else:
        image = Image.open(os.path.join(args.img_dir, img_name)).convert('RGB')

    ori_size = image.size
    # Load the annotation for big object,
    # The appearance of the smaller object is hard to preserve in the latent space of stable diffusion
    anns = coco_obj.getAnnIds(imgIds=[imid], areaRng=[64 ** 2, 32 ** 5], iscrowd=None)
    anns = coco_obj.loadAnns(anns)

    if anns:
        for r in range(args.repeat_run):
            init_img, mask_img, crop_size, init_size, updated_ann_list = get_ovtrack_data_generation_input(
                img_info, image, coco_obj)

            if init_img is None:
                save_original_image(image, f"{img_name[:-4]}_v{r}.jpg", out_dataset)
                continue

            g_img = generate_transformed_image(ovtrack_gen_pipe, prompt, init_img, mask_img, args.delta, crop_size, ori_size, guidance_scale)
            save_generated_image(g_img, f"{img_name[:-4]}_v{r}.jpg", out_dataset)

            update_final_annotations(updated_ann_list, f"{img_name[:-4]}_v{r}.jpg", final_anno_list)
    else:
        for r in range(args.repeat_run):
            save_original_image(image, f"{img_name[:-4]}_v{r}.jpg", out_dataset)

def get_caption_and_guidance_scale(cap_json, imid, imid_caption_mapping):
    if cap_json:
        cap_list = imid_caption_mapping[imid]
        idx = np.random.randint(len(cap_list))
        return cap_list[idx]['caption'], 7.5
    else:
        return "", 0

def save_original_image(image, image_name, out_dataset):
    success, encoded_image = cv2.imencode('.jpg', np.array(image)[:, :, ::-1])
    im_binary = np.asarray(encoded_image.tobytes())
    out_dataset.create_dataset(image_name, data=im_binary)

def generate_transformed_image(ovtrack_gen_pipe,
                               prompt,
                               init_img,
                               mask_img,
                               delta,
                               crop_size,
                               ori_size,
                               guidance_scale):
    with autocast(True):
        g_img = ovtrack_gen_pipe(prompt=prompt,
                             init_image=init_img,
                             mask_image=mask_img,
                             strength=delta,
                             num_inference_steps=50,
                             guidance_scale=guidance_scale,
                             n_sample=1,
                             jump_length=1,
                             jump_n_sample=1).images

    g_img = g_img[0].crop((0, 0, crop_size[0], crop_size[1]))
    return g_img.resize(ori_size)

def save_generated_image(g_img, image_name, out_dataset):
    success, encoded_image = cv2.imencode('.jpg', np.array(g_img)[:, :, ::-1])
    im_binary = np.asarray(encoded_image.tobytes())
    out_dataset.create_dataset(image_name, data=im_binary)

def update_final_annotations(updated_ann_list, image_name, final_anno_list):
    for ann in updated_ann_list:
        ann['image_name'] = image_name
        final_anno_list.append(ann)  # Assuming final_anno_list is accessible in this context


def load_repaint_pipeline(model_path, device):
    """
    Load the Stable Diffusion repaint pipeline.
    :param model_path: Path to the model.
    :param device: Device to use ('cuda' or 'cpu').
    :return: Repaint pipeline object.
    """
    # Dummy implementation - replace with actual model loading code
    return StableDiffusionOVTrackPipeline.from_pretrained(model_path, revision="fp16", torch_dtype=torch.float16,
                                                          use_auth_token=False).to(device)

def load_coco_data(coco_json_path, cap_json_path):
    """
    Load COCO data and caption mappings.
    :param coco_json_path: Path to the COCO JSON file.
    :param cap_json_path: Path to the caption JSON file.
    :return: Tuple of COCO object, image info mapping, and caption mapping.
    """
    coco_obj = COCO(coco_json_path)
    imid_iminfo_mapping = {item['id']: item for item in coco_obj.loadImgs(coco_obj.getImgIds())}

    imid_caption_mapping = {}
    if cap_json_path:
        with open(cap_json_path, 'r') as f:
            cap_data = json.load(f)
            for item in cap_data['annotations']:
                imid_caption_mapping.setdefault(item['image_id'], []).append(item)

    return coco_obj, imid_iminfo_mapping, imid_caption_mapping


def get_partitioned_image_ids(imid_iminfo_mapping, partition_index, total_partitions):
    """
    Partition the image IDs based on the specified partition index and total partitions.
    :param imid_iminfo_mapping: Mapping of image IDs to image info.
    :param partition_index: The index of the current partition.
    :param total_partitions: Total number of partitions.
    :return: List of image IDs for the specified partition.
    """
    image_ids = list(imid_iminfo_mapping.keys())
    partition_size = len(image_ids) // total_partitions
    start = partition_index * partition_size
    end = None if partition_index == total_partitions - 1 else start + partition_size
    return image_ids[start:end]


def save_annotations(pkl_out_path, annotations):
    """
    Save the annotations to a pickle file.
    :param pkl_out_path: Path to the output pickle file.
    :param annotations: Annotations to save.
    """
    # Dummy implementation - replace with actual annotation saving code
    with open(pkl_out_path, 'wb') as f:
        pickle.dump(annotations, f)


def parse_arguments():
    parser = argparse.ArgumentParser(description='OVTrack Image generation with Stable Diffusion Model.')

    parser.add_argument('--p', type=int, default=0,
                        help='Partition index to process. Used for dividing the dataset into multiple parts.')
    parser.add_argument('--total_p', type=int, default=1,
                        help='Total number of partitions.')
    parser.add_argument('--repeat_run', type=int, default=1,
                        help='Number of times to repeat the generation for each image.')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Strength of diffusion process.')
    parser.add_argument('--coco_json', type=str,
                        default='data/lvis/annotations/lvisv0.5+coco_train.json',
                        help='Path to the COCO format JSON file.')
    parser.add_argument('--h5_img', type=str,
                        help='Path to the h5 file containing images.')
    parser.add_argument('--img_dir', type=str,
                        default='data/lvis/train2017/',
                        help='Directory containing the images.')
    parser.add_argument('--cap_json', type=str,
                        help='Path to the JSON file containing captions.')
    parser.add_argument('--h5_out', type=str,
                        help='Output path for the h5 file containing generated images.')
    parser.add_argument('--pkl_out', type=str,
                        help='Output path for the pickle file containing generated annotations.')

    return parser.parse_args()



def main():

    args = parse_arguments()

    img_client = h5py.File(args.h5_img, "r") if args.h5_img else None
    out_dataset = h5py.File(args.h5_out, "a")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ovtrack_gen_pipe = load_repaint_pipeline(model_id_or_path, device)
    coco_obj, imid_iminfo_mapping, imid_caption_mapping = load_coco_data(args.coco_json, args.cap_json)
    final_anno_list = []
    for imid in get_partitioned_image_ids(imid_iminfo_mapping, args.p, args.total_p):
        process_image(imid, args, coco_obj, ovtrack_gen_pipe, out_dataset,
                      imid_iminfo_mapping,
                      imid_caption_mapping,
                      img_client,
                      final_anno_list)

    out_dataset.close()
    save_annotations(args.pkl_out, final_anno_list)
    print('Done')

if __name__ == '__main__':
    main()