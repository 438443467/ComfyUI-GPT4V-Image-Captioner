import re
import base64
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
from io import BytesIO
import requests
import io
import traceback
from PIL import Image, ExifTags

def log(message:str, message_type:str='info'):
    name = 'GPTCaptioner'
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# {name} -> {message}")



target_resolutions = [
    (640, 1632),  # 640 * 1632 = 1044480
    (704, 1472),  # 704 * 1472 = 1036288
    (768, 1360),  # 768 * 1360 = 1044480
    (832, 1248),  # 832 * 1248 = 1038336
    (896, 1152),
    (960, 1088),  # 960 * 1088 = 1044480
    (992, 1056),  # 992 * 1056 = 1047552
    (1024, 1024),  # 1024 * 1024 = 1048576
    (1056, 992),  # 1056 * 992 = 1047552
    (1088, 960),  # 1088 * 960 = 1044480
    (1152, 896),
    (1248, 832),  # 1248 * 832 = 1038336
    (1360, 768),  # 1360 * 768 = 1044480
    (1472, 704),  # 1472 * 704 = 1036288
    (1632, 640),  # 1632 * 640 = 1044480
    # (768, 1360),   # 768 * 1360 = 1044480
    # (1472, 704),   # 1472 * 704 = 1036288
    # (1024, 1024),  # 1024 * 1024 = 1048576
]


# 该函数将根据图像的EXIF方向旋转图像
def apply_exif_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()

        if exif is not None:
            exif = dict(exif.items())
            orientation_value = exif.get(orientation)

            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        # cases: image don't have getexif
        pass

    return image
def process_image(base64_image):

    try:

        img_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_data))
        img = apply_exif_orientation(img)  # Apply the EXIF orientation

        # Convert to 'RGB' if it is 'RGBA' or any other mode
        img = img.convert('RGB')

        # 计算原图像的宽高比
        original_aspect_ratio = img.width / img.height

        # 找到最接近原图像宽高比的目标分辨率
        target_resolution = min(target_resolutions, key=lambda res: abs(original_aspect_ratio - res[0] / res[1]))

        # 计算新的维度
        if img.width / target_resolution[0] < img.height / target_resolution[1]:
            new_width = target_resolution[0]
            new_height = int(img.height * target_resolution[0] / img.width)
        else:
            new_height = target_resolution[1]
            new_width = int(img.width * target_resolution[1] / img.height)

        # 等比缩放图像
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # 计算裁剪的区域
        left = int((img.width - target_resolution[0]) / 2)
        top = int((img.height - target_resolution[1]) / 2)
        right = int((img.width + target_resolution[0]) / 2)
        bottom = int((img.height + target_resolution[1]) / 2)

        # 裁剪图像
        img = img.crop((left, top, right, bottom))

        # 转换并保存图像为JPG格式
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        return base64_image

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return None

# 剔除caption中包含在exclude_words中的单词的函数
def remove_exclude_words(caption, exclude_words):
    exclude_words_list = [word.strip(" ,，") for word in exclude_words.split(",")]
    pattern = r'\b(?:{})\b'.format('|'.join(exclude_words_list))
    clean_caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
    clean_caption = re.sub(r'(, )+', ', ', clean_caption)
    return clean_caption.strip(', ')
    
# 添加提示词
def add_words_to_caption(caption, add_words):
    # Add a comma after the caption and append the additional words
    updated_caption = caption + ', ' + add_words
    
    return updated_caption

    
# 为提示词增加权重
def add_weight_to_prompt(prompt, weight):
    prompt = prompt.rstrip(',')
    formatted_weight = "{:.2f}".format(weight)
    weighted_prompt = f"{prompt}={formatted_weight}"
    return f"({weighted_prompt})"

class DalleImage:

    @staticmethod
    def tensor_to_base64(tensor: torch.Tensor) -> str:
        try:
            """
            将 PyTorch 张量转换为 base64 编码的图像。
    
            注意：ComfyUI 提供的图像张量格式为 [N, H, W, C]。
            例如，形状为 torch.Size([1, 1024, 1024, 3])
    
            参数:
                tensor (torch.Tensor): 要转换的图像张量。
    
            返回:
                str: base64 编码的图像字符串。
            """
            # 将张量转换为 PIL 图像
            if tensor.ndim == 4:
                tensor = tensor.squeeze(0)  # 如果存在批量维度，则移除
            pil_image = Image.fromarray((tensor.numpy() * 255).astype('uint8'))

            # 将 PIL 图像保存到缓冲区
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")  # 可以根据需要更改为 JPEG 格式
            buffer.seek(0)

            # 将缓冲区编码为 base64
            base64_image = base64.b64encode(buffer.read()).decode('utf-8')

            return base64_image

        except Exception as e:
            print(f"Error in tensor_to_base64: {e}")
            traceback.print_exc()
            return f"Error in tensor_to_base64: {e}"

# 初始化缓存变量
cached_result = None
cached_seed = None
cached_image = None
cached_full_caption = None
cached_custom_prompt = None

class GPTCaptioner:
    def __init__(self):
        self.saved_api_key = ''
        self.saved_exclude_words = ''

    #定义输入参数类型和默认值
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "api_url": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"max": 0xffffffffffffffff, "min": 1, "step": 1, "default": 1, "display": "number"}),
                "model": (["gpt-4-vision-preview", "gpt-4o", "gpt-4o-ca", "gpt-4-turbo"], {"default": "gpt-4o"}),
                "img_quality": (["auto", "high", "low"], {"default": "auto"}),
                "timeout": ("INT", {"max": 0xffffffffffffffff, "min": 1, "step": 1, "default": 30, "display": "number"}),
                "enable_weight": ("BOOLEAN", {"default": False}),
                "weight" : ("FLOAT", {"max": 8.201, "min": 0.0, "step": 0.1, "display": "number", "round": 0.01, "default": 1}),
                "custom_prompt": ("STRING",
                                   {
                                       "default": "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image.",
                                       "multiline": True, "dynamicPrompts": False
                                   }),
                "exclude_words": ("STRING",
                                   {
                                       "default": "",
                                       "multiline": True, "dynamicPrompts": False
                                   }),
                "add_words": ("STRING",
                                   {
                                       "default": "",
                                       "multiline": True, "dynamicPrompts": False
                                   }),
                "image": ("IMAGE", {})
            }
        }

    #定义输出名称、类型、所属位置
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("prompt","full_prompt",)
    FUNCTION = "gogo"
    OUTPUT_NODE = False
    CATEGORY = "Sanmi Simple Nodes/GPT"

    # 对排除词进行处理
    @staticmethod
    def clean_response_text(text: str) -> str:
        try:
            cleaned_text = text.replace("，", ",").replace("。", "")
            cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()
            return cleaned_text

        except Exception as e:
            print(f"Error in clean_response_text: {e}")
            traceback.print_exc()
            return f"Error in clean_response_text: {e}"


    # 调用 OpenAI API，将图像和文本提示发送给 API 并获取生成的文本描述，处理可能出现的异常情况，并返回相应的结果或错误信息。
    @staticmethod
    def run_openai_api(api_key, api_url, seed, model, img_quality, timeout, custom_prompt, exclude_words, add_words, image):
        global cached_result, cached_seed, cached_image, cached_full_caption
        # 判断seed值、image值、提示词、模型是否发生变化
        if (cached_seed is not None and cached_image is not None and cached_seed == seed and cached_image == image):
            caption = cached_result
            caption = remove_exclude_words(caption, exclude_words)
            caption = add_words_to_caption(caption, add_words)
            full_caption = cached_full_caption
            return (caption,full_caption,)

        prompt = "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image."
        image_base64 = image
        if custom_prompt != prompt:
            prompt = custom_prompt
        data = {
            "model": f"{model}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": f"{img_quality}"
                        }
                         }
                    ]
                }
            ],
            "max_tokens": 300
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 配置重试策略
        retries = Retry(total=5,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"])  # 更新参数名
        #处理可能发生的请求异常，并返回相应的错误消息。
        with requests.Session() as s:
            s.mount('https://', HTTPAdapter(max_retries=retries))

            try:
                response = s.post(api_url, headers=headers, json=data, timeout=timeout)
                response.raise_for_status()  # 如果请求失败，将抛出 HTTPError
            except requests.exceptions.HTTPError as errh:
                return f"HTTP Error: {errh}"
            except requests.exceptions.ConnectionError as errc:
                return f"Error Connecting: {errc}"
            except requests.exceptions.Timeout as errt:
                return f"Timeout Error: {errt}"
            except requests.exceptions.RequestException as err:
                return f"OOps: Something Else: {err}"

        try:
            response_data = response.json()

            if 'error' in response_data:
                return f"API error: {response_data['error']['message']}"

            caption = response_data["choices"][0]["message"]["content"]
            full_caption = caption
            # 更新缓存变量
            cached_result = caption
            cached_seed = seed
            cached_image = image
            cached_full_caption = full_caption
            cached_custom_prompt = custom_prompt

            # 剔除caption中包含在exclude_words中的单词
            caption = remove_exclude_words(caption, exclude_words)
            # 增加提示词
            caption = add_words_to_caption(caption, add_words)

            return (caption,full_caption,)
        except Exception as e:
            return (f"Failed to parse the API response: {e}\n{response.text}",None)

    # 根据用户输入的参数构建指令，并使用 GPT 模型进行请求，返回相应的结果。将之前的值进行转换
    def gogo(self, api_key, api_url, seed, model, img_quality, timeout, enable_weight, weight, custom_prompt, exclude_words, add_words, image):
        try:

            # 如果 image 是 torch.Tensor 类型，则将其转换为 base64 编码的图像
            if isinstance(image, torch.Tensor):
                image = DalleImage.tensor_to_base64(image)

            # 自动裁切base64 编码图像
            image = process_image(image)

            # 请求 prompt
            result = self.run_openai_api(api_key, api_url, seed, model, img_quality, timeout, custom_prompt, exclude_words, add_words, image)

            # 检查解包结果是否正确
            if len(result) != 2:
                log(f"Error unpacking result: {result}", "error")
                caption = result
                full_caption = result
            else:
                caption, full_caption = result
            # 给 prompt 增加权重
            if enable_weight:
                caption = add_weight_to_prompt(caption, weight)

            return (caption,full_caption,)
        except Exception as e:
            print(f"Error in gogo: {e}")
            traceback.print_exc()
            return (f"Error in gogo: {e}",None,)

#定义功能和模块名称
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "GPT4VCaptioner": GPTCaptioner,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT4VCaptioner": "GPT4V-Image-Captioner",
}







