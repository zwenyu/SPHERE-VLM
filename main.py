import os, json
import argparse
from eval_datasets.coco_dataset import COCODataset
from models import get_model_by_name
from tasks.space_vqa import SpaceVQAEvaluator

def main(args):

    # Load dataset
    dataset = COCODataset(args.annotations_dir, args.annotations_json, args.save_suffix, args.img_dir)

    # Load model
    model = get_model_by_name(args.model_name, load=(not args.eval_saved_predictions))

    # Perform valuation task
    evaluator = SpaceVQAEvaluator(model, dataset, args.num_seeds, args)
    accuracy, validity = evaluator.evaluate(eval_saved_predictions=args.eval_saved_predictions)
    print(f"Accuracy Score: {accuracy}, Validity Score: {validity}")

    # Save results
    save_path = os.path.join(args.annotations_dir, f"{args.results_filename}.json")
    if os.path.exists(save_path):
        with open(save_path, 'r') as json_file:
            results = json.load(json_file)
    else:
        results = {}

    annotations_json = args.annotations_json if args.eval_qn_type == "all" \
        else f"{args.annotations_json}-{args.eval_qn_type}"
    if annotations_json not in results:
        results[annotations_json] = {"accuracy": {}, "validity": {}}

    results[annotations_json]["accuracy"][args.model_name] = accuracy
    results[annotations_json]["validity"][args.model_name] = validity

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="instruct_blip")
    parser.add_argument("--annotations_dir", type=str,
                        default="eval_datasets/coco_test2017_annotations")
    parser.add_argument("--annotations_json", type=str,
                        default="single_skill/size_only")
    parser.add_argument("--img_dir", type=str,
                        default="eval_datasets/coco_test2017")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_suffix", type=str, default="w_predictions")
    parser.add_argument("--results_filename", type=str, default="results")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--eval_saved_predictions", action="store_true")
    parser.add_argument("--eval_qn_type", type=str, default="all", 
        choices=["all", "intermediate", "final", "allo", "ego"])    
    args = parser.parse_args()

    main(args)

