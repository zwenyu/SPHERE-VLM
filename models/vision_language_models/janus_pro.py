"""
Janus-Pro: https://huggingface.co/deepseek-ai/Janus-Pro-7B, https://github.com/deepseek-ai/Janus
"""

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import torch
from models.model_interface import VisionLanguageModel


class JanusPro(VisionLanguageModel):
    def __init__(self, device="cuda", load=True):
        self.device = device
        self.model_name = "janus_pro"
        self.model_path = "deepseek-ai/Janus-Pro-7B"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval().to(self.device)

    def predict(self, image, image_path, text):

        # process image and text
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{text} Answer the question directly.",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        inputs = self.processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)

        # autoregressively complete prompt
        output = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
        )
        text_response = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return text_response
