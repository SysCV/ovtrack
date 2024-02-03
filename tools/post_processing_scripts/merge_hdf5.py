
import argparse
import h5py
import json
import os
import tqdm
import pickle

def main(h5_folder_name, coco_lvis_path, total_p, repeat_num, default_save_dir):

    # Specify the final h5 file name for the combined file.
    final_client_h5_path = os.path.join(default_save_dir, h5_folder_name + '.h5')
    final_client = h5py.File(final_client_h5_path, "a")
    print('Save final h5 file to: ', final_client_h5_path)

    # image groundtruth json
    coco_lvis = json.load(open(coco_lvis_path))

    # partition the groups following the creation rule.
    image_id_list = [item['id'] for item in coco_lvis['images']]
    imid_iminfo_mapping = {}
    for item in coco_lvis['images']:
        if item['id'] not in imid_iminfo_mapping:
            imid_iminfo_mapping[item['id']] = item

    im_num_total = len(image_id_list)
    img_per_partition = im_num_total // total_p
    groups = {}
    for p in range(total_p):
        if p == total_p - 1:
            groups[p] = image_id_list[p * img_per_partition:]
        else:
            groups[p] = image_id_list[p * img_per_partition:p * img_per_partition + img_per_partition]

    # Create the final h5 file client which stores all the generated images.
    for key in groups:
        group_imid_list = groups[key]
        img_client = h5py.File(f'{default_save_dir}/{h5_folder_name}/{h5_folder_name}_total_p_{total_p}_split_{key}.h5', "r")
        for imid in tqdm.tqdm(group_imid_list):
            imname = imid_iminfo_mapping[imid]['file_name']
            tmp_imname_list = [imname]
            for j in range(repeat_num):
                tmp_imname_list.append(f'{imname[:-4]}_v{j}.jpg')
            for imname in tmp_imname_list:
                data = img_client[imname]
                final_client.create_dataset(imname, data=data)
        img_client.close()
    final_client.close()

    # Create the final pkl file which stores all annotations.
    updated_anno_list = []
    for key in groups:
        anno_list = pickle.load(open(
            f'{default_save_dir}/{h5_folder_name}/{h5_folder_name}_total_p_{total_p}_split_{key}.pkl','rb'))
        updated_anno_list += anno_list

    pickle.dump(updated_anno_list, open(f'{default_save_dir}/{h5_folder_name}.pkl', 'wb'))
    print('Save final pkl file to: ', f'{default_save_dir}/{h5_folder_name}.pkl')
    print('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge the generated hdf5 image partitions.')
    parser.add_argument('--h5_folder_name', type=str, default='ovtrack_gen_results', help='The folder name which saves the generated partitions')
    parser.add_argument('--coco_lvis_path', type=str, default='data/lvis/annotations/lvisv1_coco_10_base.json', help='Path to the the merged coco-lvis json')
    parser.add_argument('--total_p', type=int, default=8, help='Total partitions')
    parser.add_argument('--default_save_dir', type=str, default="./data/tao/ovtrack/" , help='Default save directory for hdf5 partition folders and final .pkl and .h5 files. ')
    parser.add_argument('--repeat_num', type=int, default=1, help='Repeat number')

    args = parser.parse_args()
    main(args.h5_folder_name, args.coco_lvis_path, args.total_p, args.repeat_num, args.default_save_dir)
