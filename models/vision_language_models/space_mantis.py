"""
SpaceMantis: https://huggingface.co/remyxai/SpaceMantis
"""

import torch
from models.model_interface import VisionLanguageModel
from models.utils.SpaceMantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava


class SpaceMantis(VisionLanguageModel):
    def __init__(self, device="cuda", load=True):
        self.device = device
        self.model_name = "space_mantis"
        self.model_path = "remyxai/SpaceMantis"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.processor = MLlavaProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation=None,
            low_cpu_mem_usage=True
        )
        self.model.eval().to(self.device)

    def predict(self, image, image_path, text):

        generation_kwargs = {
            "max_new_tokens": 100,
            "num_beams": 1,
            "do_sample": False
        }

        text_response, history = chat_mllava(
            f"{text} Answer the question directly.", [image], self.model, self.processor,
            **generation_kwargs
        )

        return text_response
