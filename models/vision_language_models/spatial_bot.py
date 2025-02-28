"""
SpatialBot: https://huggingface.co/RussRobin/SpatialBot-3B
ZoeDepth estimation: https://huggingface.co/Intel/zoedepth-kitti [for depth estimation]
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from models.model_interface import VisionLanguageModel


class SpatialBot(VisionLanguageModel):
    def __init__(self, device="cuda", load=True):
        self.device = device
        self.model_name = "spatial_bot"
        self.model_path = "RussRobin/SpatialBot-3B"
        self.depth_model_path = "Intel/zoedepth-nyu-kitti"
        if load:
            self.load_model(model_path=self.model_path)
            self.depth_estimator = pipeline(task="depth-estimation", model=self.depth_model_path)

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
            USER: <image 1>\n<image 2>\n{text} Answer the question directly. ASSISTANT:"
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in query.split("<image 1>\n<image 2>\n")]
        input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:],
            dtype=torch.long).unsqueeze(0).to(self.device)

        # depth estimation
        depth_outputs = self.depth_estimator(image)
        depth_map = depth_outputs["depth"]

        channels = len(depth_map.getbands())
        if channels == 1:
            img = np.array(depth_map)
            height, width = img.shape
            three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
            three_channel_array[:, :, 0] = (img // 1024) * 4
            three_channel_array[:, :, 1] = (img // 32) * 8
            three_channel_array[:, :, 2] = (img % 32) * 8
            depth_map = Image.fromarray(three_channel_array, 'RGB')

        image_tensor = self.model.process_images([image, depth_map], self.model.config).to(
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
