#!/bin/bash

# Check if an argument is provided for total_p
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_batch.sh <total_p> <h5_folder_name>"
    exit 1
fi

# Set the total_p from the argument
total_p=$1 # total number of partitions, it should be the same as the number of GPUs.
h5_folder_name=$2 # create a folder to store all the generated images files.

echo "Running batch process..."
echo "Total processes: $total_p"
echo "H5 folder name: $h5_folder_name"

default_save_dir="./data/tao/ovtrack/" # run from the root folder
echo "Default save directory: ${default_save_dir}"

# Create a directory to store log files
mkdir -p logs/${h5_folder_name}
echo "Log directory created at: logs/${h5_folder_name}"

mkdir -p ${default_save_dir}/${h5_folder_name}
echo "Data directory created at: ${default_save_dir}/${h5_folder_name}"

# Iterate from 0 to total_p-1
for i in $(seq 0 $(($total_p - 1)))
do
  echo "Starting process $i..."
  # Set CUDA_VISIBLE_DEVICES for this iteration and run the command in the background
  CUDA_VISIBLE_DEVICES=${i} python3 -m diffusers_clean.ovtrack_diffusion_generation \
    --p ${i} \
    --total_p ${total_p} \
    --repeat_run 1 \
    --delta 0.75 \
    --coco_json ./data/lvis/annotations/lvisv1_coco_10_base.json \
    --h5_img ./data/lvis/train_imgs.hdf5 \
    --cap_json ./data/lvis/annotations/captions_train2017.json \
    --h5_out ${default_save_dir}/${h5_folder_name}/${h5_folder_name}_total_p_${total_p}_split_${i}.h5 \
    --pkl_out ${default_save_dir}/${h5_folder_name}/${h5_folder_name}_total_p_${total_p}_split_${i}.pkl \
    > logs/${h5_folder_name}/${h5_folder_name}_total_p_${total_p}_split_${i}.log &
  echo "Process $i running in background..."
done

echo "All processes started. Waiting for completion..."
# Wait for all background jobs to finish
wait
echo "Batch processing completed."