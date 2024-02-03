import argparse
import copy
import h5py
import io
import json
import math
import os
import pickle
import random
import torch
from mmdet.datasets.pipelines.transforms import Pad, Resize
from PIL import Image
from pycocotools.coco import COCO
# make sure you're logged in with `huggingface-cli login`
from torch.cuda.amp import autocast
import numpy as np
from .pipelines.stable_diffusion.pipeline_ovtrack_image_generation import StableDiffusionOVTrackPipeline

### for debugging purposes ######
import random
import torch

def set_random_seed(seed_value):
    """
    Set the seed for various modules to ensure reproducibility.
    :param seed_value: An integer value for the seed.
    """
    random.seed(seed_value)  # Python's built-in random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch

    # If you are using CUDA with PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set a fixed seed for reproducibility
SEED = 42
set_random_seed(SEED)
##################################

device = "cuda"
# replace with your own path
# model_id_or_path  = "/cluster/work/cvl/lisiyu/stable-diffusion-v1-4/"
# model_id_or_path  = "/scratch/snx3000/rxiang/dl/work/stable-diffusion-v1-4"
model_id_or_path  = "/scratch/lisiyu/stable-diffusion-v1-4/"


def _get_rotation_matrix(rotate_degrees):
   radian = math.radians(rotate_degrees)
   rotation_matrix = np.array(
       [[np.cos(radian), -np.sin(radian), 0.],
        [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
       dtype=np.float32)
   return rotation_matrix


def _get_scaling_matrix(scale_ratio):
   scaling_matrix = np.array(
       [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
       dtype=np.float32)
   return scaling_matrix


def _get_share_matrix(scale_ratio):
   scaling_matrix = np.array(
       [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
       dtype=np.float32)
   return scaling_matrix


def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
   x_radian = math.radians(x_shear_degrees)
   y_radian = math.radians(y_shear_degrees)
   shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                            [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                           dtype=np.float32)
   return shear_matrix


def _get_translation_matrix(x, y):
   translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                 dtype=np.float32)
   return translation_matrix


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
   rotation_matrix = _get_rotation_matrix(rotation_degree)

   # Scaling
   scaling_ratio = random.uniform(scaling_ratio_range[0],
                                  scaling_ratio_range[1])
   scaling_matrix = _get_scaling_matrix(scaling_ratio)

   # Shear
   x_degree = random.uniform(-max_shear_degree,
                             max_shear_degree)
   y_degree = random.uniform(-max_shear_degree,
                             max_shear_degree)
   shear_matrix = _get_shear_matrix(x_degree, y_degree)

   # Translation
   trans_x = random.uniform(-max_translate_ratio,
                            max_translate_ratio) * width
   trans_y = random.uniform(-max_translate_ratio,
                            max_translate_ratio) * height
   translate_matrix = _get_translation_matrix(trans_x, trans_y)

   warp_matrix = (
           translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)

   return warp_matrix


def compute_ioa(mask1, mask2):
   # mask1 is the target
   intersection = np.logical_and(mask1, mask2).sum()
   area1 = mask1.sum()
   return intersection / area1


def update_anno(annos, anno_mask_list):
   new_annos = []
   for i, ann in enumerate(annos):
       new_ann = copy.deepcopy(ann)
       ann_mask = anno_mask_list[i]

       bboxes = extract_bboxes(ann_mask[:, :, np.newaxis])
       new_ann['bbox'] = list(bboxes[0])
       new_ann['area'] = ann_mask.sum()
       new_annos.append(new_ann)

   return new_annos


def merge_mask(mask_list):
   if mask_list:
       mask = mask_list[0]
       if len(mask_list) > 1:
           for t_m in mask_list[1:]:
               mask = np.logical_or(mask, t_m)
       return mask
   else:
       return []

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


import numpy as np
import cv2
from PIL import Image
import albumentations as A


def get_ovtrack_data_generation_input(
       iminfo,
       pil_I,
       coco_obj,
       area_range_min=64 ** 2,  # Min area for annotations
       area_range_max=32 ** 5,  # Max area for annotations
       affine_scale=(0.6, 1.5),  # Scale range for affine transformation
       affine_rotate=(-30, 30),  # Rotation range for affine transformation
       affine_translate=(0, 0.2),  # Translation range for affine transformation
       affine_shear=(-15, 15),  # Shear range for affine transformation
       perspective_scale=(0.05, 0.2),  # Scale for perspective transformation
       flip_probability=0.5,  # Probability for horizontal flip
       brightness_contrast_probability=0.2  # Probability for random brightness/contrast
):
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

   # Filter annotations based on area range
   annIds = coco_obj.getAnnIds(imgIds=[imid], areaRng=[area_range_min, area_range_max], iscrowd=None)
   anns = coco_obj.loadAnns(annIds)

   # Remove crowd annotations
   non_crowd_anns = [a for a in anns if a.get('iscrowd', 0) == 0]
   anns = non_crowd_anns

   if not anns:
       return None, None, None, None, None

   # Sort annotations by area in descending order
   sorted_anns = sorted(anns, key=lambda item: item['area'], reverse=True)

   # Generate masks for annotations
   mask_dict = {ann['id']: coco_obj.annToMask(ann) for ann in sorted_anns}

   # Define random geometric transform with color contrast
   transform = A.Compose([
       A.Affine(scale=affine_scale, rotate=affine_rotate, translate_percent=affine_translate, shear=affine_shear,
                p=0.5, mode=0),
       A.Perspective(scale=perspective_scale, keep_size=True, p=0.5),
       A.HorizontalFlip(p=flip_probability),
       A.RandomBrightnessContrast(p=brightness_contrast_probability),
   ])

   # Apply transformations
   mask_list = list(mask_dict.values())
   transformed = transform(image=I, masks=mask_list)
   transformed_image = transformed['image']
   transformed_mask = transformed['masks']
   updated_ann_list = update_anno(sorted_anns, transformed_mask)  # Assuming update_anno is a defined function

   # Merge original and transformed images
   out_mask = (transformed_image == 0)
   new_img = (I * out_mask + transformed_image * (1 - out_mask)).astype(np.uint8)

   # Process mask
   mask_f = merge_mask(transformed_mask).astype(np.uint8)  # Assuming merge_mask is a defined function
   kernel = np.ones((15, 15), np.uint8)
   new_mask = cv2.dilate(mask_f, kernel, iterations=2)

   # Resize and pad image and mask
   resize_img_with_ration = Resize(img_scale=[(768, 512)], keep_ratio=True)  # Assuming Resize is a defined function
   pad_32 = Pad(size_divisor=64)  # Assuming Pad is a defined function
   init_mask = Image.fromarray(new_mask * 255)
   init_img = Image.fromarray(new_img)
   init_size = init_img.size

   results = {'img': np.array(init_img)}
   mask_results = {'img': np.array(init_mask)}
   results = resize_img_with_ration(results)
   results = pad_32(results)
   mask_results = resize_img_with_ration(mask_results)
   mask_results = pad_32(mask_results)

   init_img = Image.fromarray(results['img'])
   mask_img = Image.fromarray(mask_results['img'])
   crop_size = tuple(np.rint(results['scale_factor'][:2] * init_size))

   return init_img, mask_img, crop_size, init_size, updated_ann_list


def parse_args():

   parser = argparse.ArgumentParser(description='qdtrack test model')
   parser.add_argument('--p', default=0, type=int, help='choose which partitons')
   parser.add_argument('--total_p', default=1, type=int, help='set the number of partisions')
   parser.add_argument('--repeat_run', default=1,type=int, help='repeat generation of the same image')
   parser.add_argument('--delta',default=0.75, type=float, help='how many times the diffusion sample from the same image')
   parser.add_argument('--coco_json', default='data/lvis/annotations/lvisv0.5+coco_train.json'
                       ,help='specify the partition that the model belong')
   parser.add_argument('--h5_img',
                       help='specify the partition that the model belong')
   parser.add_argument('--img_dir', default='data/lvis/train2017/',
                       help='specify the partition that the model belong')
   parser.add_argument('--cap_json',
                       help='specify the partition that the model belong')
   parser.add_argument('--h5_out', help='output result h5 file which saves generated images e.g. :data/lvis/mini_lvis_val_images.hdf5')
   parser.add_argument('--pkl_out', help='output result pkl file which saves the generated annotation  e.g. :data/lvis/mini_lvis_val_images.pkl')
   args = parser.parse_args()

   return args


def main():

   args = parse_args()
   if args.h5_img:
       img_client = h5py.File(args.h5_img, "r")
   out_dataset = h5py.File(args.h5_out, "a")

   repaint_pipe = StableDiffusionOVTrackPipeline.from_pretrained(
       model_id_or_path,
       revision="fp16",
   torch_dtype=torch.float16, use_auth_token=False)
   repaint_pipe = repaint_pipe.to(device)
   lvis_v05_train_coco = json.load(
       open(args.coco_json, 'r')
   )

   #load COCO
   lvis = COCO(args.coco_json)

   if args.cap_json:
       cap_2017 = json.load(open(args.cap_json))
       imid_caption_mapping = {}
       for item in cap_2017['annotations']:
           if item['image_id'] not in imid_caption_mapping:
               imid_caption_mapping[item['image_id']] = [item]
           else:
               imid_caption_mapping[item['image_id']].append(item)

   image_id_list = [item['id'] for item in lvis_v05_train_coco['images']]

   imid_iminfo_mapping = {}
   for item in lvis_v05_train_coco['images']:
       if item['id'] not in imid_iminfo_mapping:
           imid_iminfo_mapping[item['id']] = item

   im_num_total = len(image_id_list)
   img_per_partition = im_num_total // args.total_p
   groups = {}
   for p in range(args.total_p):
       if p == args.total_p - 1:
           groups[p] = image_id_list[p * img_per_partition:]
       else:
           groups[p] = image_id_list[p * img_per_partition:p * img_per_partition + img_per_partition]

   p_img_list = groups[args.p]

   final_anno_list = []
   for imid in p_img_list:
       img_info = imid_iminfo_mapping[imid]
       img_name = img_info['file_name']

       if args.cap_json:
           cap_list = imid_caption_mapping[imid]
           idx = np.random.randint(len(cap_list))
           prompt = cap_list[idx]['caption']
           guidance_scale = 7.5
       else:
           prompt=""
           guidance_scale = 0

       # define image resize function
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
       annIds = lvis.getAnnIds(imgIds=[imid], areaRng=[64 ** 2, 32 ** 5], iscrowd=None)
       anns = lvis.loadAnns(annIds)
       if anns:
           for r in range(args.repeat_run):
               init_img, mask_img, crop_size, \
               init_size, updated_ann_list = get_ovtrack_data_generation_input(img_info,
                                                                               pil_I=image,
                                                                            coco_obj=lvis)

               if init_img is None:
                   success, encoded_image = cv2.imencode('.jpg', np.array(image)[:, :, ::-1])
                   im_binary = np.asarray(encoded_image.tobytes())
                   out_dataset.create_dataset(f"{img_name[:-4]}_v{r}.jpg", data=im_binary)
                   continue

               with autocast(True):
                   g_img = repaint_pipe(prompt=prompt,
                                        init_image=init_img,
                                        mask_image = mask_img,
                                        strength=args.delta,
                                        num_inference_steps=50,
                                        guidance_scale=guidance_scale,
                                        n_sample=1,
                                        jump_length=1,
                                        jump_n_sample=1,
                                        ).images

               g_img = g_img[0].crop((0, 0, crop_size[0], crop_size[1]))
               g_img = g_img.resize(ori_size)
               success, encoded_image = cv2.imencode('.jpg', np.array(g_img)[:, :, ::-1])
               im_binary = np.asarray(encoded_image.tobytes())
               out_dataset.create_dataset(f"{img_name[:-4]}_v{r}.jpg", data=im_binary)
               for ann in updated_ann_list:
                   ann['image_name'] = f"{img_name[:-4]}_v{r}.jpg"
                   final_anno_list.append(ann)

       else:
           for r in range(args.repeat_run):
               success, encoded_image = cv2.imencode('.jpg', np.array(image)[:, :, ::-1])
               im_binary = np.asarray(encoded_image.tobytes())
               out_dataset.create_dataset(f"{img_name[:-4]}_v{r}.jpg", data=im_binary)

   out_dataset.close()
   pickle.dump(final_anno_list, open(args.pkl_out, 'wb'))
   print('Done')

if __name__ == '__main__':
   main()

