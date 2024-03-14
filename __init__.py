from .GPT4V_captioner import NODE_CLASS_MAPPINGS as styClassMappings, NODE_DISPLAY_NAME_MAPPINGS as styDisplay
from .Load_Image_Batch import NODE_CLASS_MAPPINGS as utilClassMappings, NODE_DISPLAY_NAME_MAPPINGS as utilDisplay


NODE_CLASS_MAPPINGS = styClassMappings
# 定义模块的公开接口
NODE_CLASS_MAPPINGS = {**styClassMappings, **utilClassMappings}

NODE_DISPLAY_NAME_MAPPINGS = {**styDisplay, **utilDisplay}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']



