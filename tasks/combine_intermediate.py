"""
Create .json files for reasoning tasks by adding intermediate questions and their answers
to the main question.
"""

import os, json
import re
import random

class CombineIntermediate:
    def __init__(self, dataset, save_suffix=""):
        self.dataset = dataset
        self.save_suffix = save_suffix

    def combine_questions(self):

        question_map = {}

        # Group questions by `question_id` (main part) and subpart
        for _, _, _, ann in self.dataset:
            question_id_main = ann["question_id"].split(".")[0]

            # Initialize the main question entry if not already present
            if question_id_main not in question_map:
                question_map[question_id_main] = ann
            else:
                # Merge question and options
                main_entry = question_map[question_id_main]
                # shuffle options
                options = ann["option"]
                random.shuffle(options)
                query = " or ".join(options).capitalize() + "?"
                text = " ".join([ann["question"], query])
                intermediate_info = \
                    f"Given that for the question: {text} The answer is: {ann['answer']}.\n"
                main_entry["question"] = intermediate_info + main_entry["question"]

        # Save combined questions
        combined_ann = list(question_map.values())
        save_path = os.path.join(
            self.dataset.annotations_dir,
            f"{self.dataset.annotations_json}_{self.save_suffix}.json"
        )
        with open(save_path, 'w') as f:
            json.dump(combined_ann, f, indent=4)
