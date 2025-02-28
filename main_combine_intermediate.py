import os, json
import argparse
from eval_datasets.coco_dataset import COCODataset
from tasks.combine_intermediate import CombineIntermediate

def main(args):

    # Load dataset
    dataset = COCODataset(args.annotations_dir, args.annotations_json, args.save_suffix, args.img_dir,
        raise_error=False)

    # Combine intermediate questions
    combine = CombineIntermediate(dataset, args.save_suffix)
    combine.combine_questions()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_dir", type=str,
                        default="eval_datasets/coco_test2017_annotations")
    parser.add_argument("--annotations_json", type=str,
                        default="reasoning/object_manipulation")
    parser.add_argument("--img_dir", type=str,
                        default="eval_datasets/coco_test2017")
    parser.add_argument("--save_suffix", type=str, default="w_intermediate")
    args = parser.parse_args()

    main(args)

