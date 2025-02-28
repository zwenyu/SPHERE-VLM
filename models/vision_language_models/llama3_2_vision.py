"""
Llama-3.2-Vision: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
"""

import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from models.model_interface import VisionLanguageModel


class Llama3_2_Vision(VisionLanguageModel):
    def __init__(self, device="cuda", version="11B", load=True):
        self.device = device
        self.model_name = f"llama3_2_vision_{version.lower()}"
        self.model_path = f"meta-llama/Llama-3.2-{version}-Vision-Instruct"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.model.eval()

    def predict(self, image, image_path, text):

        # process image and text
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{text} Answer the question directly."},
                    {"type": "image"},
                    ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        inputs.update({
            "max_new_tokens": 100,
        })

        # autoregressively complete prompt
        output = self.model.generate(**inputs)
        text_output = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )
        text_response = text_output.split("assistant\n")[1].strip()

        return text_response
