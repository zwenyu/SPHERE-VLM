"""
LLaVA-OneVision: https://huggingface.co/docs/transformers/en/model_doc/llava_onevision
"""

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from models.model_interface import VisionLanguageModel


class LlavaOneVision(VisionLanguageModel):
    def __init__(self, device="cuda", load=True):
        self.device = device
        self.model_name = "llava_onevision"
        self.model_path = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval().to(self.device)

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
        ).to(self.device, torch.float16)
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
