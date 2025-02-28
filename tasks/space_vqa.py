import os, json
import random
import numpy as np
from tqdm import tqdm
from .utils import set_seed, process_format_num, process_format_mcq, clean_string
from .metrics import compute_accuracy, compute_validity

class SpaceVQAEvaluator:
    def __init__(self, model, dataset, num_seeds, args):
        self.model = model
        self.dataset = dataset
        self.num_seeds = num_seeds
        self.save_predictions = args.save_predictions
        self.save_suffix = args.save_suffix
        self.eval_qn_type = args.eval_qn_type

    def evaluate(self, eval_saved_predictions=False):
        correct = 0
        valid = 0
        total = 0

        if self.save_predictions:
            save_path = os.path.join(
                self.dataset.annotations_dir,
                f"{self.dataset.annotations_json}_{self.save_suffix}.json"
            )
            dataset_w_predictions = self.dataset.coco_annotations

        for idx, image, image_path, ann in tqdm(self.dataset,
            desc="evaluation",
            total=self.dataset.__len__()):

            if self.eval_qn_type == "final" and "." in ann["question_id"]:
                continue
            if self.eval_qn_type == "intermediate" and "." not in ann["question_id"]:
                continue
            if self.eval_qn_type == "allo" and ann["metadata"]["viewpoint"] != "allo":
                continue
            if self.eval_qn_type == "ego" and ann["metadata"]["viewpoint"] != "ego":
                continue

            if self.model == "baseline":

                if ann["metadata"]["format"] == "num":
                    correct_i = 0.0
                else:
                    correct_i = 1.0 / len(ann["option"])
                valid_i = 0.0

            else:
                if "raw_prediction" not in ann:
                    ann["raw_prediction"] = {}
                    ann["prediction"] = {}
                    ann["valid"] = {}
                    ann["correct"] = {}
                if eval_saved_predictions or (self.model.model_name in ann["raw_prediction"]):  # skips if prediction exists
                    raw_prediction_list = ann["raw_prediction"][self.model.model_name]
                else:
                    raw_prediction_list = []

                    for seed in range(self.num_seeds):
                        set_seed(seed)
                        if ann["metadata"]["format"] == "num":
                            query = "Give the answer in number format."
                        else:
                            # shuffle options
                            options = ann["option"]
                            random.shuffle(options)
                            query = " or ".join(options).capitalize() + "?"
                        text = " ".join([ann["question"], query])

                        # raw prediction
                        raw_prediction = self.model.predict(
                            image=image,
                            image_path=image_path,
                            text=text
                        )
                        raw_prediction_list.append(raw_prediction)

                prediction_list = []
                for seed in range(self.num_seeds):
                    if ann["metadata"]["format"] == "num":
                        query = "Give the answer in number format."
                    else:
                        # shuffle options
                        options = ann["option"]
                        query = " or ".join(options).capitalize() + "?"
                    text = " ".join([ann["question"], query])                    
                    raw_prediction = raw_prediction_list[seed]

                    # processed prediction
                    if ann["metadata"]["format"] == "num":
                        prediction = process_format_num(clean_string(raw_prediction))
                    else:
                        prediction = process_format_mcq(raw_prediction, ann["option"], text)
                    prediction_list.append(prediction)

                answer = clean_string(ann["answer"])
                valid_i = np.mean([p != "-" for p in prediction_list])
                correct_i = np.mean([answer == p for p in prediction_list])  # invalid answers are counted as wrong

            correct += correct_i
            valid += valid_i
            total += 1

            if self.save_predictions:
                ann["raw_prediction"][self.model.model_name] = raw_prediction_list
                ann["prediction"][self.model.model_name] = prediction_list
                ann["valid"][self.model.model_name] = valid_i
                ann["correct"][self.model.model_name] = correct_i
                dataset_w_predictions[idx] = ann

            if self.save_predictions and total % 10 == 0:
                with open(save_path, "w") as f:
                    json.dump(dataset_w_predictions, f, indent=4)

        if self.save_predictions:
            with open(save_path, "w") as f:
                json.dump(dataset_w_predictions, f, indent=4)

        accuracy = compute_accuracy(correct, total)
        validity = compute_validity(valid, total)
        return accuracy, validity

