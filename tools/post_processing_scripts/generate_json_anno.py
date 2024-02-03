
import argparse
import json
import pickle
import copy

def main(h5_folder_name, coco_lvis_path, repeat_num, default_save_dir):

    # image groundtruth json
    coco_lvis = json.load(open(coco_lvis_path))

    # load final pkl annotation file from the combined files.
    generated_anno = pickle.load(open(f'{default_save_dir}/{h5_folder_name}.pkl','rb'))

    fake_imname_anno_mapping = {}
    for ann in generated_anno:
        ann['area'] = int(ann['area'])
        ann['bbox'] = [int(b) for b in ann['bbox']]
        if ann['image_name'] not in fake_imname_anno_mapping:
            fake_imname_anno_mapping[ann['image_name']] = [ann]
        else:
            fake_imname_anno_mapping[ann['image_name']].append(ann)

    imid_iminfo_mapping = {}
    for item in coco_lvis['images']:
        if item['id'] not in imid_iminfo_mapping:
            imid_iminfo_mapping[item['id']] = item
    imid_anno_mapping = {}
    for item in coco_lvis['annotations']:
        imid = item['image_id']
        if imid not in imid_anno_mapping:
            imid_anno_mapping[imid] = [item]
        else:
            imid_anno_mapping[imid].append(item)

    new_dataset = {}
    new_dataset['images'] = []
    new_dataset['annotations'] = []
    new_dataset['videos'] = []
    new_dataset['categories'] = copy.deepcopy(coco_lvis['categories'])

    imid_list = [item['id'] for item in coco_lvis['images']]
    annoid_list = [item['id'] for item in coco_lvis['annotations']]
    start_imid = max(imid_list)
    start_annoid = max(annoid_list)

    ## adding static images as videos
    video_id = 0
    track_id = 0
    for im in coco_lvis['images']:
        video_dict = {}
        video_dict['id'] = video_id
        video_dict['width'] = im['width']
        video_dict['height'] = im['height']
        new_dataset['videos'].append(video_dict)
        f_id = 0
        imname = im['file_name']
        new_im = copy.deepcopy(im)
        new_im['video_id'] = video_id
        new_im['frame_id'] = f_id
        new_dataset['images'].append(new_im)
        new_ann_list = []
        if im['id'] in imid_anno_mapping:
            anno_list = imid_anno_mapping[im['id']]
            for ann in anno_list:
                if 'iscrowd' in ann and ann['iscrowd'] == 1:
                    continue
                else:
                    new_ann = copy.deepcopy(ann)
                    new_ann['instance_id'] = track_id
                    new_ann['track_id'] = track_id
                    new_ann['video_id'] = video_id
                    new_ann_list.append(new_ann)
                    track_id += 1

        new_dataset['annotations'] += new_ann_list

        video_id += 1

    # adding generated images as videos
    for im in coco_lvis['images']:
        video_dict = {}
        video_dict['id'] = video_id
        video_dict['width'] = im['width']
        video_dict['height'] = im['height']
        new_dataset['videos'].append(video_dict)
        f_id = 0
        imname = im['file_name']
        new_im = copy.deepcopy(im)
        start_imid += 1
        ori_imid = im['id']
        new_im['id'] = start_imid
        new_im['video_id'] = video_id
        new_im['frame_id'] = f_id
        new_dataset['images'].append(new_im)
        new_ann_list = []
        if ori_imid in imid_anno_mapping:
            anno_list = imid_anno_mapping[im['id']]
            for ann in anno_list:
                # skip crowd annoations
                if 'iscrowd' in ann and ann['iscrowd'] == 1:
                    continue
                # if the object area is too small, the stable diffusion won't work due to its latent structure.
                # We select 64** based on heuristic
                if ann['area'] > 64 ** 2:
                    new_ann = copy.deepcopy(ann)
                    ori_ann_id = ann['id']
                    start_annoid += 1
                    new_ann['id'] = start_annoid
                    new_ann['ori_ann_id'] = ori_ann_id
                    new_ann['image_id'] = start_imid
                    new_ann['instance_id'] = track_id
                    new_ann['track_id'] = track_id
                    new_ann['video_id'] = video_id
                    new_ann_list.append(new_ann)
                    track_id += 1

        new_dataset['annotations'] += new_ann_list
        for j in range(repeat_num):
            start_imid += 1
            f_id += 1
            im_dict = copy.deepcopy(new_im)
            im_dict['file_name'] = f'{imname[:-4]}_v{j}.jpg'
            im_dict['id'] = start_imid
            im_dict['frame_id'] = f_id
            im_dict['video_id'] = video_id
            new_dataset['images'].append(im_dict)

            if ori_imid in imid_anno_mapping:
                if im_dict['file_name'] in fake_imname_anno_mapping:
                    fake_annos = fake_imname_anno_mapping[im_dict['file_name']]
                    fake_aid_annos_mapping = {}
                    for fake_a in fake_annos:
                        if fake_a['id'] not in fake_aid_annos_mapping:
                            fake_aid_annos_mapping[fake_a['id']] = fake_a

                    for ann in new_ann_list:
                        start_annoid += 1
                        ori_ann_id = ann['ori_ann_id']
                        if ori_ann_id in fake_aid_annos_mapping:
                            new_ann = copy.deepcopy(fake_aid_annos_mapping[ori_ann_id])
                            if new_ann['bbox'][2] < 1 or new_ann['bbox'][3] < 1 or new_ann['area'] < 32 * 32:
                                continue

                            new_ann['id'] = start_annoid
                            new_ann['image_id'] = start_imid
                            new_ann['instance_id'] = ann['track_id']
                            new_ann['track_id'] = ann['track_id']
                            new_ann['video_id'] = ann['video_id']
                            del new_ann['segmentation']
                            new_dataset['annotations'].append(new_ann)
                else:

                    for ann in new_ann_list:
                        start_annoid += 1
                        new_ann = copy.deepcopy(ann)
                        new_ann['id'] = start_annoid
                        new_ann['image_id'] = start_imid
                        new_dataset['annotations'].append(new_ann)

        video_id += 1

    final_json_annotation_path = f'./data/tao/annotations/ovtrack/{h5_folder_name}.json'
    print('Saving annotations to {}'.format(final_json_annotation_path))
    json.dump(new_dataset, open(final_json_annotation_path, 'w'))
    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge the generated hdf5 image partitions.')
    parser.add_argument('--h5_folder_name', type=str, default='The folder name which saves the generated partitions', help='H5 folder name')
    parser.add_argument('--coco_lvis_path', type=str, default='data/lvis/annotations/lvis_v1_train+coco_mask_v1_base.json', help='Path to the the merged coco-lvis json')
    parser.add_argument('--default_save_dir', type=str, default="./data/tao/ovtrack/" , help='Default save directory for hdf5 partition folders and final .pkl and .h5 files. ')
    parser.add_argument('--repeat_num', type=int, default=1, help='Repeat number')

    args = parser.parse_args()
    main(args.h5_folder_name, args.coco_lvis_path, args.repeat_num, args.default_save_dir)
