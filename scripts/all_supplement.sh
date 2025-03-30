#!/bin/bash

cd /home/zhang_wenyu/Project/SPHERE-VLM
export PYTHONPATH=/home/zhang_wenyu/Project/SPHERE-VLM/models/utils/SpatialRGPT:$PYTHONPATH

models=(
    "gemini-2.0-flash-exp"
    "gpt-4o"
    "idefics2"
    "instruct_blip"
    "llava_next"
    "llava_onevision"
    "phi_3.5_vision"
    "intern_vl2_5"
    "qwen_vl"
    "qwen2_vl_7b"
    "qwen2_vl_72b"
    "llama3_2_vision_11b"
    "llama3_2_vision_90b"
    "spatialrgpt_rgb"
    "space_mantis"
    "spatial_bot_rgb"
    "spatial_bot"
)

single_skill_json=(position_only)
combine_2_skill_json=(position_and_counting)
reasoning_json=(object_manipulation object_occlusion)

for model_name in "${models[@]}"; do
    echo ${model_name}
    if [[ ${model_name} == "gemini-2.0-flash-exp" || ${model_name} == "gpt-4o" ]]; then
        num_seeds=1
    else
        num_seeds=5
    fi

    for file in ${single_skill_json[@]}; do
        echo ${file}
        python main.py --model_name ${model_name} --annotations_json single_skill/${file} --num_seeds ${num_seeds} --eval_saved_predictions --eval_qn_type allo
        python main.py --model_name ${model_name} --annotations_json single_skill/${file} --num_seeds ${num_seeds} --eval_saved_predictions --eval_qn_type ego
    done

    for file in ${combine_2_skill_json[@]}; do
        echo ${file}
        python main.py --model_name ${model_name} --annotations_json combine_2_skill/${file} --num_seeds ${num_seeds} --eval_saved_predictions --eval_qn_type allo
        python main.py --model_name ${model_name} --annotations_json combine_2_skill/${file} --num_seeds ${num_seeds} --eval_saved_predictions --eval_qn_type ego
    done

    for file in ${reasoning_json[@]}; do
        echo ${file}
        python main.py --model_name ${model_name} --annotations_json reasoning/${file} --num_seeds ${num_seeds} --eval_saved_predictions --eval_qn_type intermediate
        python main.py --model_name ${model_name} --annotations_json reasoning/${file} --num_seeds ${num_seeds} --eval_saved_predictions --eval_qn_type final
    done
done

echo "All scripts executed successfully."
