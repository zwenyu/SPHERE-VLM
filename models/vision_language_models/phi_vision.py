"""
Phi-3.5-vision: https://huggingface.co/microsoft/Phi-3.5-vision-instruct
"""

from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from models.model_interface import VisionLanguageModel


class PhiVision(VisionLanguageModel):
    def __init__(self, device="cuda", load=True):
        self.device = device
        self.model_name = "phi_3.5_vision"
        self.model_path = "microsoft/Phi-3.5-vision-instruct"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            num_crops=4
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2"
        )
        self.model.eval().to(self.device)

    def predict(self, image, image_path, text):

        prompt = f"<|user|>\n<|image_1|>\n{text} Answer the question directly.<|end|>\n<|assistant|>\n"

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        inputs.update({
            "max_new_tokens": 100,
            "temperature": 0.0,
            "do_sample": False,
            "eos_token_id": self.processor.tokenizer.eos_token_id
        })

        # autoregressively complete prompt
        output = self.model.generate(**inputs)
        generate_ids = output[:, inputs["input_ids"].shape[1]:]
        text_response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        return text_response
