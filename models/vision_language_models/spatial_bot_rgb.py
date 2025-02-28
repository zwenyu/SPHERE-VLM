"""
SpatialBot: https://huggingface.co/RussRobin/SpatialBot-3B
This version uses RGB images as input, without depth map.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from models.model_interface import VisionLanguageModel


class SpatialBot_RGB(VisionLanguageModel):
    def __init__(self, device="cuda", load=True):
        self.device = device
        self.model_name = "spatial_bot_rgb"
        self.model_path = "RussRobin/SpatialBot-3B"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval().to(self.device)

    def predict(self, image, image_path, text):

        offset_bos = 0
        query = f"A chat between a curious user and an artificial intelligence assistant. \
            The assistant gives helpful, detailed, and polite answers to the user's questions. \
            USER: <image 1>\n{text} Answer the question directly. ASSISTANT:"
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in query.split("<image 1>\n")]
        input_ids = torch.tensor(text_chunks[0] + [-201] + text_chunks[1][offset_bos:],
            dtype=torch.long).unsqueeze(0).to(self.device)

        image_tensor = self.model.process_images([image], self.model.config).to(
            dtype=self.model.dtype, device=self.device)
        self.model.get_vision_tower().to(self.device)

        # generate
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=100,
            use_cache=True,
            repetition_penalty=1.0  # increase this to avoid chattering
        )[0]
        text_response = self.tokenizer.decode(
            output_ids[input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return text_response
