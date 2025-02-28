"""
Google Gemini

gemini-2.0-flash-exp: 10 RPM, 1500 RPD
"""

import os
import torch
import time
from models.model_interface import VisionLanguageModel
import google.generativeai as genai


class GeminiAPI(VisionLanguageModel):
    def __init__(self, device="cuda", version="gemini-1.5-flash", load=True):
        self.device = device
        self.model_name = version
        self.model_path = version
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(model_path)

    def predict(self, image, image_path, text):

        prompt = [image, f"{text} Answer the question directly."]
        text_output = self.model.generate_content(prompt).text
        text_response = text_output.strip()

        time.sleep(10)

        return text_response
