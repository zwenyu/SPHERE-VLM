from .vision_language_models.llava_next import LlavaNext
from .vision_language_models.llava_onevision import LlavaOneVision
from .vision_language_models.idefics2 import Idefics2
from .vision_language_models.idefics3 import Idefics3
from .vision_language_models.instruct_blip import InstructBLIP
from .vision_language_models.phi_vision import PhiVision
from .vision_language_models.qwen_vl import QwenVL
from .vision_language_models.qwen2_vl import Qwen2VL
from .vision_language_models.qwen2_5_vl import Qwen2_5_VL
from .vision_language_models.intern_vl2_5 import InternVL2_5
from .vision_language_models.llama3_2_vision import Llama3_2_Vision
# from .vision_language_models.janus_pro import JanusPro
# from .vision_language_models.spatialrgpt_rgb import SpatialRGPT_RGB
# from .vision_language_models.space_mantis import SpaceMantis
from .vision_language_models.spatial_bot_rgb import SpatialBot_RGB
from .vision_language_models.spatial_bot import SpatialBot
from .vision_language_models.gemini import GeminiAPI
from .vision_language_models.gpt import OpenAIAPI

def get_model_by_name(model_name, device="cuda", load=True):
    if model_name == "baseline":
        return "baseline"
    elif model_name == "llava_next":
        return LlavaNext(device=device, load=load)
    elif model_name == "llava_onevision":
        return LlavaOneVision(device=device, load=load)
    elif model_name == "idefics2":
        return Idefics2(device=device, load=load)
    elif model_name == "idefics3":
        return Idefics3(device=device, load=load)
    elif model_name == "instruct_blip":
        return InstructBLIP(device=device, load=load)
    elif model_name == "phi_3.5_vision":
        return PhiVision(device=device, load=load)
    elif model_name == "qwen_vl":
        return QwenVL(device=device, load=load)
    elif model_name == "qwen2_vl_7b":
        return Qwen2VL(device=device, version="7B", load=load)
    elif model_name == "qwen2_vl_72b":
        return Qwen2VL(device=device, version="72B", load=load)
    elif model_name == "qwen2_5_vl_7b":
        return Qwen2_5_VL(device=device, version="7B", load=load)
    elif model_name == "qwen2_5_vl_72b":
        return Qwen2_5_VL(device=device, version="72B", load=load)
    elif model_name == "intern_vl2_5":
        return InternVL2_5(device=device, load=load)
    elif model_name == "llama3_2_vision_11b":
        return Llama3_2_Vision(device=device, version="11B", load=load)
    elif model_name == "llama3_2_vision_90b":
        return Llama3_2_Vision(device=device, version="90B", load=load)
    # elif model_name == "janus_pro":
    #     return JanusPro(device=device, load=load)
    # elif model_name == "spatialrgpt_rgb":
    #     return SpatialRGPT_RGB(device=device, load=load)
    # elif model_name == "space_mantis":
    #     return SpaceMantis(device=device, load=load)
    elif model_name == "spatial_bot_rgb":
        return SpatialBot_RGB(device=device, load=load)
    elif model_name == "spatial_bot":
        return SpatialBot(device=device, load=load)
    elif model_name[:6] == "gemini":
        return GeminiAPI(version=model_name, load=load)
    elif model_name[:3] == "gpt" or model_name[:2] == "o4":
        return OpenAIAPI(version=model_name, load=load)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

