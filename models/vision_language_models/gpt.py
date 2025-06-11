"""
OpenAI GPT

gpt-4o-mini: 500 RPM, 10,000 RPD
gpt-4o: 500 RPM
o4-mini: 500 RPM
"""

import os
import torch
import time
import base64
from models.model_interface import VisionLanguageModel
from openai import OpenAI


class OpenAIAPI(VisionLanguageModel):
    def __init__(self, device="cuda", version="gpt-4o-mini", load=True):
        self.device = device
        self.model_name = version
        self.model_path = version
        if load:
            self.load_model(model_path=self.model_path)

    def load_model(self, model_path=None):
        self.client = OpenAI()

    def predict(self, image, image_path, text):

        prompt = f"{text} Answer the question directly."
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )

        text_response = response.choices[0].message.content

        time.sleep(2)

        return text_response


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")