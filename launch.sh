#!/bin/bash

# Define your conda environment and configuration path variables
conda_goodtg="/data/shurui.gui/mambaforge/envs/ATTA/bin/python -m ATTA.kernel.alg_main"
auto_args_config_root="TTA_configs"

# Define the array of 'k' values
#k_values=(0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3)
k_values=(1 2 3 4 5 6)


# Get the segment number from the command line argument
segment=$1
seg_size=$2
gpu_idx=$3
dataset=$4
if [[ -z "$segment" || $segment -lt 1 || $segment -gt $seg_size ]]; then
    echo "Error: Segment number must be between 1 and $seg_size."
    exit 1
fi

# Calculate the number of elements per segment
total_elements=${#k_values[@]}
elements_per_segment=$((total_elements / $seg_size))

# Calculate the start and end indices for the segment
start_index=$((($segment - 1) * elements_per_segment))
end_index=$(($start_index + $elements_per_segment - 1))

# If it's the last segment, include any remaining elements
if [ $segment -eq $seg_size ]; then
    end_index=$((total_elements - 1))
fi

# Loop over the specified ranges and conditions for the segment
for k_idx in $(seq $start_index $end_index); do
    k=${k_values[$k_idx]}
    for le in 0 1; do
        for ic in 0; do     # Original: 0 1;
            # Check if ic is 0 and k is not an integer
            if [[ $ic -eq 0 && $(echo "$k" | awk '{print $1%1}') != 0 ]]; then
                continue
            fi

            # if dataset is PACS then el is 1e-4, otherwise 1e-3
            if [[ $dataset == "PACS" ]]; then
                el=1e-4
            else
                el=1e-3
            fi

            # Construct the command
            cmd="${conda_goodtg} --task train --config_path ${auto_args_config_root}/${dataset}/SimATTA.yaml --atta.SimATTA.cold_start 100 \
                --atta.SimATTA.el ${el} --atta.SimATTA.nc_increase $k --atta.gpu_clustering --exp_round 1 --atta.SimATTA.LE $le \
                --atta.SimATTA.target_cluster $ic --log_file SimATTA_${dataset}_LE${le}_IC${ic}_k${k}_el${el} --num_workers 4 --gpu_idx $gpu_idx"

            # Execute the command
            echo "Executing: $cmd"
            eval $cmd
        done
    done
done

