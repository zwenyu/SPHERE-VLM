"""
InstructBLIP: https://huggingface.co/Salesforce/instructblip-vicuna-7b
"""

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from models.model_interface import VisionLanguageModel


class InstructBLIP(VisionLanguageModel):
    def __init__(self, device="cuda", load=True):
        self.device = device
        self.model_name = "instruct_blip"
        self.model_path = "Salesforce/instructblip-vicuna-7b"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.processor = InstructBlipProcessor.from_pretrained(
            model_path
        )
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval().to(self.device)

    def predict(self, image, image_path, text):

        inputs = self.processor(
            images=image,
            text=f"{text} Answer the question directly.",
            return_tensors="pt"
        ).to(self.device)
        inputs.update({
            "do_sample": False,
            "num_beams": 5,
            "max_length": 256,
            "min_length": 1,
            "top_p": 0.9,
            "repetition_penalty": 1.5,
            "length_penalty": 1.0,
            "temperature": 1
        })

        # autoregressively complete prompt
        output = self.model.generate(**inputs)
        text_response = self.processor.decode(output[0], skip_special_tokens=True)

        return text_response
