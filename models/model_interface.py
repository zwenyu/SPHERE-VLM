from abc import ABC, abstractmethod

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

class VisionLanguageModel(ABC):

    @abstractmethod
    def load_model(self, model_path):
        """Loads the model from the specified path."""
        pass

    @abstractmethod
    def predict(self, image, question, option, format):
        """Returns the model's predictions."""
        pass
