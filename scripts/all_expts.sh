#!/bin/bash

scripts=(
    "./run_baseline.sh"
    "./gemini_flash.sh"
    "./run_gpt4o.sh"
    "./run_idefics2.sh"
    "./run_instruct_blip.sh"
    "./run_llava_next.sh"
    "./run_llava_onevision.sh"
    "./run_phi_vision.sh"
    "./run_intern_vl2_5.sh"
    "./run_qwen_vl.sh"
    "./run_qwen2_vl_7b.sh"
    "./run_qwen2_vl_72b.sh"
    "./run_qwen2_5_vl_7b.sh"
    "./run_qwen2_5_vl_72b.sh"
    "./run_janus_pro.sh"
    "./run_llama3_2_vision_11b.sh"
    "./run_llama3_2_vision_90b.sh"
    './run_spatialrgpt.sh'
    "./run_space_mantis.sh"
    "./run_spatial_bot_rgb.sh"
    "./run_spatial_bot.sh"
)

# Loop through each script and execute it
for script in "${scripts[@]}"
do
    if [ -f "$script" ]; then  # Check if the script file exists
        echo "Running $script..."
        bash "$script"
        if [ $? -ne 0 ]; then  # Check if the script ran successfully
            echo "Error: $script failed."
            exit 1
        fi
    else
        echo "Error: $script not found."
        exit 1
    fi
done

echo "All scripts executed successfully."
