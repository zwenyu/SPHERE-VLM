import os, json
import argparse
import numpy as np
from eval_datasets.coco_dataset import COCODataset

def main(args):

    # Load results
    results_path = os.path.join(args.annotations_dir, f"{args.results_json}.json")
    with open(results_path) as f:
        results = json.load(f)

    single_skill = [
        'single_skill/position_only',
        'single_skill/counting_only-paired-position_and_counting',
        'single_skill/counting_only-paired-distance_and_counting',
        'single_skill/distance_only',
        'single_skill/size_only'
    ]
    multi_skill = [
        'combine_2_skill/position_and_counting',
        'combine_2_skill/distance_and_counting',
        'combine_2_skill/distance_and_size'
    ]
    reasoning = [
        'reasoning/object_occlusion-final',
        'reasoning/object_manipulation-final'
    ]

    # extract values and find average
    values_single_skill_all = []
    for key in single_skill:
        try:
            values_single_skill_all.append(results[key][args.metric][args.model_name])
        except KeyError:
            values_single_skill_all.append(0.0)
    # combine two counting values
    values_single_skill = [
        values_single_skill_all[0],
        (values_single_skill_all[1] + values_single_skill_all[2]) / 2,
        values_single_skill_all[3],
        values_single_skill_all[4]
    ]
    values_single_skill.append(np.mean(values_single_skill))

    values_multi_skill = []
    for key in multi_skill:
        try:
            values_multi_skill.append(results[key][args.metric][args.model_name])
        except KeyError:
            values_multi_skill.append(0.0)
    values_multi_skill.append(np.mean(values_multi_skill))

    values_reasoning = []
    for key in reasoning:
        try:
            values_reasoning.append(results[key][args.metric][args.model_name])
        except KeyError:
            values_reasoning.append(0.0)
    values_reasoning.append(np.mean(values_reasoning))

    values = values_single_skill + values_multi_skill + values_reasoning
    values.append(np.mean(values))
    values_str = " & ".join([f"{v*100:.1f}" for v in values])
    print(values_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_dir", type=str,
                        default="eval_datasets/coco_test2017_annotations")
    parser.add_argument("--results_json", type=str,
                        default="results")
    parser.add_argument("--metric", type=str,
                        default="accuracy", choices=["accuracy", "validity"])
    parser.add_argument("--model_name", type=str,
                        default="instruct_blip")
    args = parser.parse_args()

    main(args)

