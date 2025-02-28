"""
SpatialRGPT: https://github.com/AnjieCheng/SpatialRGPT
Strategy with depth map is not used as highly accurate region proposals are required.
"""

import torch
import os
from models.model_interface import VisionLanguageModel
from models.utils.SpatialRGPT.llava.model.builder import load_pretrained_model
from models.utils.SpatialRGPT.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from models.utils.SpatialRGPT.llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from models.utils.SpatialRGPT.llava.conversation import conv_templates

class SpatialRGPT_RGB(VisionLanguageModel):
    def __init__(self, device="cuda", load=True):
        self.device = device
        self.model_name = "spatialrgpt_rgb"
        self.model_path = "a8cheng/SpatialRGPT-VILA1.5-8B"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        model_path = os.path.expanduser(self.model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, model_name, model_base=None)
        self.model.eval().to(self.device)

    def predict(self, image, image_path, text):

        # process image and text
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        qs += " Answer the question directly."

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        image_tensor = self.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0].unsqueeze(0).to(self.device, torch.float16)

        # autoregressively complete prompt
        output = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=None,
            top_p=None,
            num_beams=1,
            max_new_tokens=100,
            use_cache=True,
        )
        text_output = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        text_response = text_output.strip()

        return text_response
