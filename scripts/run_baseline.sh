#!/bin/bash

cd /home/zhang_wenyu/Project/SPHERE

model_name=baseline
single_skill_json=(size_only distance_only position_only counting_only-paired-distance_and_counting counting_only-paired-position_and_counting)
combine_2_skill_json=(distance_and_size distance_and_counting position_and_counting)
reasoning_json=(object_manipulation object_occlusion object_manipulation_w_intermediate object_occlusion_w_intermediate)

for file in ${single_skill_json[@]}; do
    echo ${file}
    python main.py --model_name ${model_name} --annotations_json single_skill/${file}
done

for file in ${combine_2_skill_json[@]}; do
    echo ${file}
    python main.py --model_name ${model_name} --annotations_json combine_2_skill/${file}
done

for file in ${reasoning_json[@]}; do
    echo ${file}
    python main.py --model_name ${model_name} --annotations_json reasoning/${file}
done