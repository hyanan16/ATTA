#!/bin/bash

# Define the name of the tmux session
dataset=$2
session_name="exps_${dataset}"

# Path to your script
script_path="./launch.sh"
seg_size=$1
IFS=',' read -r -a gpu_idx <<< $GPU_IDX

# Start a new tmux session
tmux new-session -d -s $session_name

# Create windows for each segment
for i in $(seq 1 $seg_size); do
    if [ $i -eq 1 ]; then
        # For the first segment, rename the first window rather than creating a new one
        tmux rename-window -t $session_name:1 "Segment $i"
        tmux send-keys -t $session_name:1 "$script_path $i $seg_size ${gpu_idx[i-1]} ${dataset}" C-m
    else
        # For other segments, create new windows
        tmux new-window -t $session_name -n "Segment $i"
        tmux send-keys -t $session_name:"Segment $i" "$script_path $i $seg_size ${gpu_idx[i-1]} ${dataset}" C-m
    fi
done

# Attach to the tmux session
#tmux attach-session -t $session_name
