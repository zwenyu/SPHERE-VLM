"""
Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
"""

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from models.model_interface import VisionLanguageModel
from qwen_vl_utils import process_vision_info


class Qwen2_5_VL(VisionLanguageModel):
    def __init__(self, device="cuda", version="7B", load=True):
        self.device = device
        self.model_name = f"qwen2_5_vl_{version.lower()}"
        self.model_path = f"Qwen/Qwen2.5-VL-{version}-Instruct"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
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
