"""
Qwen-VL: https://huggingface.co/Qwen/Qwen-VL-Chat
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.model_interface import VisionLanguageModel


class QwenVL(VisionLanguageModel):
    def __init__(self, device="cuda", load=True):
        self.device = device
        self.model_name = "qwen_vl"
        self.model_path = "Qwen/Qwen-VL-Chat"
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            fp16=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval().to(self.device)

    def predict(self, image, image_path, text):

        # Either a local path or an url between <img></img> tags.
        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": f"{text} Answer the question directly."},
        ])

        text_response, history = self.model.chat(self.tokenizer, query=query, history=None)

        return text_response
