# By WASasquatch (Discord: WAS#0263)
#
# Copyright 2023 Jordan Thompson (WASasquatch)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#import cv2
#from deepface import DeepFace

from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
from typing import Optional, Union, List
from urllib.request import urlopen
from datetime import datetime
from comfy_extras.chainner_models import model_loading
from comfy.model_management import InterruptProcessingException
from comfy.cli_args import args
from nodes import PreviewImage, SaveImage
from numba import jit
from tqdm import tqdm
from server import PromptServer
from aiohttp import web
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image as PilImage, ImageEnhance

import pandas as pd
import latent_preview
import piexif
import piexif.helper
import comfy.diffusers_convert
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.clip_vision
import comfy.model_management
import cv2
import folder_paths as comfy_paths
import ast
import glob
import hashlib
import json
import nodes
import math
import numpy as np
import os
import random
import re
import requests
import socket
import subprocess
import sys
import datetime
import time
import torch
import string
import inspect
import folder_paths



#filepath：存储数据的JSON文件的路径。
#data：从JSON文件读取的数据的字典。

#GLOBALS
#获取当前文件的绝对路径
NODE_FILE = os.path.abspath(__file__)
#NODE_FILE的父目录路径
WAS_SUITE_ROOT = os.path.dirname(NODE_FILE)
#环境变量的值
WAS_CONFIG_DIR = os.environ.get('WAS_CONFIG_DIR', WAS_SUITE_ROOT)
#was_suite_settings.json路径
WAS_DATABASE = os.path.join(WAS_CONFIG_DIR, 'was_suite_settings.json')
#was_history.json路径
WAS_HISTORY_DATABASE = os.path.join(WAS_CONFIG_DIR, 'was_history.json')
WAS_CONFIG_FILE = os.path.join(WAS_CONFIG_DIR, 'was_suite_config.json')
MODELS_DIR =  comfy_paths.models_dir
#文件扩展名的元组
ALLOWED_EXT = ('.jpeg', '.jpg', '.png',
                        '.tiff', '.gif', '.bmp', '.webp')

#! WAS SUITE CONFIG

was_conf_template = {
                    "run_requirements": True,
                    "suppress_uncomfy_warnings": True,
                    "show_startup_junk": True,
                    "show_inspiration_quote": True,
                    "text_nodes_type": "STRING",
                    "webui_styles": None,
                    "webui_styles_persistent_update": True,
                    "blip_model_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth",
                    "blip_model_vqa_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth",
                    "sam_model_vith_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "sam_model_vitl_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "sam_model_vitb_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    "history_display_limit": 36,
                    "use_legacy_ascii_text": False,
                    "ffmpeg_bin_path": "/path/to/ffmpeg",
                    "ffmpeg_extra_codecs": {
                        "avc1": ".mp4",
                        "h264": ".mkv",
                    },
                    "wildcards_path": os.path.join(WAS_SUITE_ROOT, "wildcards"),
                    "wildcard_api": True,
                }

# Create, Load, or Update Config

def getSuiteConfig():
    global was_conf_template
    try:
        with open(WAS_CONFIG_FILE, "r") as f:
            was_config = json.load(f)
    except OSError as e:
        cstr(f"Unable to load conf file at `{WAS_CONFIG_FILE}`. Using internal config template.").error.print()
        return was_conf_template
    except Exception as e:
        cstr(f"Unable to load conf file at `{WAS_CONFIG_FILE}`. Using internal config template.").error.print()
        return was_conf_template
    return was_config

def updateSuiteConfig(conf):
    try:
        with open(WAS_CONFIG_FILE, "w", encoding='utf-8') as f:
            json.dump(conf, f, indent=4)
    except OSError as e:
        print(e)
        return False
    except Exception as e:
        print(e)
        return False
    return True

if not os.path.exists(WAS_CONFIG_FILE):
    if updateSuiteConfig(was_conf_template):
        cstr(f'Created default conf file at `{WAS_CONFIG_FILE}`.').msg.print()
        was_config = getSuiteConfig()
    else:
        cstr(f"Unable to create default conf file at `{WAS_CONFIG_FILE}`. Using internal config template.").error.print()
        was_config = was_conf_template

else:
    was_config = getSuiteConfig()

    update_config = False
    for sett_ in was_conf_template.keys():
        if not was_config.__contains__(sett_):
            was_config.update({sett_: was_conf_template[sett_]})
            update_config = True

    if update_config:
        updateSuiteConfig(was_config)


# WAS Suite Locations Debug
if was_config.__contains__('show_startup_junk'):
    if was_config['show_startup_junk']:
        print(f"Running At: {NODE_FILE}")
        print(f"Running From: {WAS_SUITE_ROOT}")

# Check Write Access
if not os.access(WAS_SUITE_ROOT, os.W_OK) or not os.access(MODELS_DIR, os.W_OK):
    print(f"There is no write access to `{WAS_SUITE_ROOT}` or `{MODELS_DIR}`. Write access is required!")
    exit



# WAS SETTINGS MANAGER
class WASDatabase:
    """
    WAS Suite数据库类提供了一个简单的键值数据库，它使用JSON格式将数据存储在一个平面文件中。每个键值对都与一个类别关联。

    属性：
        filepath (str)：存储数据的JSON文件的路径。
        data (dict)：从JSON文件读取的数据的字典。

    方法：
        insert(category, key, value)：将键值对插入到指定类别的数据库中。
        get(category, key)：从数据库中检索与指定键和类别关联的值。
        update(category, key)：从数据库中更新与指定键和类别关联的值。
        delete(category, key)：从数据库中删除与指定键和类别关联的键值对。
        _save()：将数据库的当前状态保存到JSON文件中。
    """
    def __init__(self, filepath):
        self.filepath = filepath
        try:
            with open(filepath, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}
    #检查指定的类别是否存在。
    def catExists(self, category):
        return category in self.data
    #检查指定的类别和键是否存在。
    def keyExists(self, category, key):
        return category in self.data and key in self.data[category]
    #将键值对插入到指定类别的数据库中
    def insert(self, category, key, value):
        if not isinstance(category, str) or not isinstance(key, str):
            print("Category and key must be strings")
            return

        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value
        self._save()
    #更新指定类别和键关联的值。
    def update(self, category, key, value):
        if category in self.data and key in self.data[category]:
            self.data[category][key] = value
            self._save()
    #更新指定类别的所有键值对。
    def updateCat(self, category, dictionary):
        self.data[category].update(dictionary)
        self._save()
    #从数据库中检索指定类别和键关联的值。
    def get(self, category, key):
        return self.data.get(category, {}).get(key, None)
    #返回整个数据库的字典。
    def getDB(self):
        return self.data
    #插入一个新的类别。
    def insertCat(self, category):
        if not isinstance(category, str):
            print("Category must be a string")
            return

        if category in self.data:
            print(f"The database category '{category}' already exists!")
            return
        self.data[category] = {}
        self._save()
    #返回指定类别的所有键值对。
    def getDict(self, category):
        if category not in self.data:
            print(f"The database category '{category}' does not exist!")
            return {}
        return self.data[category]
    #从数据库中删除指定类别和键关联的键值对。
    def delete(self, category, key):
        if category in self.data and key in self.data[category]:
            del self.data[category][key]
            self._save()
    #将数据库的当前状态保存到JSON文件中。
    def _save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=4)
        except FileNotFoundError:
            print(f"Cannot save database to file '{self.filepath}'. "
                 "Storing the data in the object instead. Does the folder and node file have write permissions?")
        except Exception as e:
            print(f"Error while saving JSON data: {e}")

# 初始化settings数据库
WDB = WASDatabase(WAS_DATABASE)
# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

#将新的图片路径添加到历史记录中。
def update_history_images(new_paths):

    HDB = WASDatabase(WAS_HISTORY_DATABASE)
    if HDB.catExists("History") and HDB.keyExists("History", "Images"):
        saved_paths = HDB.get("History", "Images")
        for path_ in saved_paths:
            if not os.path.exists(path_):
                saved_paths.remove(path_)
        if isinstance(new_paths, str):
            if new_paths in saved_paths:
                saved_paths.remove(new_paths)
            saved_paths.append(new_paths)
        elif isinstance(new_paths, list):
            for path_ in new_paths:
                if path_ in saved_paths:
                    saved_paths.remove(path_)
                saved_paths.append(path_)
        HDB.update("History", "Images", saved_paths)
    else:
        if not HDB.catExists("History"):
            HDB.insertCat("History")
        if isinstance(new_paths, str):
            HDB.insert("History", "Images", [new_paths])
        elif isinstance(new_paths, list):
            HDB.insert("History", "Images", new_paths)
# SHA-256 Hash
def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

# LOAD IMAGE BATCH
class SAMIN_Load_Image_Batch:
    def __init__(self):
        self.HDB = WASDatabase(WAS_HISTORY_DATABASE)
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "number_image", "random"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": 'Batch 001', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '*', "multiline": False}),
                "allow_RGBA_output": (["false", "true"],),
                "rename_images": (["false", "true"],),
                "image_number": ("INT",{"default": 0, "min": 0, "max": 150000, "step": 1,"forceInput": False}),

            },
            "optional": {
                "filename_text_extension": (["true", "false"],),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING","STRING","INT",)
    RETURN_NAMES = ("image","filename_text","image_path","isTrue")
    FUNCTION = "load_batch_images"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    def load_batch_images(self, path, pattern='*', index=0, mode="single_image", label='Batch 001', allow_RGBA_output='false', rename_images='false',filename_text_extension='true', image_number=None):
        single_image_path = ''
        allow_RGBA_output = (allow_RGBA_output == 'true')

        if path == '':
            path = 'C:'
        if not os.path.exists(path):
                return (None,)

        # 创建BatchImageLoader对象并获取图像路径
        fl = self.BatchImageLoader(path, label, pattern)
        # 符合规则的图像升序的绝对路径列表
        new_paths = fl.image_paths

        if mode == 'number_image' and path != 'C:' and rename_images == 'true':
            fl.rename_images_with_sequence(path)

        # 根据加载模式选择加载图像的方式
        if mode == 'single_image':
            image, filename = fl.get_image_by_id(index)
            if image == None:
                print(f"No valid image was found for the inded `{index}`")
                return (None, None)
        if mode == 'incremental_image':
            image, filename = fl.get_next_image()
            if image == None:
                print(f"No valid image was found for the next ID. Did you remove images from the source directory?")
                return (None, None,None)
        if mode == 'number_image':
            image, filename, single_image_path ,isTrue= fl.get_image_by_number(image_number)
            if image == None:
                print(f"No valid image was found for the next ID. Did you remove images from the source directory?")
                return (self.create_black_image(), None, None, isTrue)
        else:
            newindex = int(random.random() * len(fl.image_paths))
            image, filename = fl.get_image_by_id(newindex)
            if image == None:
                print(f"No valid image was found for the next ID. Did you remove images from the source directory?")
                return (None, None)

        # 更新历史图像
        update_history_images(new_paths)

        if not allow_RGBA_output:

            image = image.convert("RGB")

        # 如果不保留文件名的文本扩展名，则去除文件名的扩展名部分
        if filename_text_extension == "false":

            filename = os.path.splitext(filename)[0]

        # 返回将图像转换为张量后的图像和文件名
        return (pil2tensor(image), filename, single_image_path, isTrue)


    class BatchImageLoader:
        def __init__(self, directory_path, label, pattern):
            # 初始化BatchImageLoader对象
            self.WDB = WDB
            self.image_paths = []
            self.load_images(directory_path, pattern)
            self.image_paths.sort()
            stored_directory_path = self.WDB.get('Batch Paths', label)
            stored_pattern = self.WDB.get('Batch Patterns', label)

            # 如果存储的路径或模式与当前路径或模式不匹配，则重置索引和存储的路径和模式
            if stored_directory_path != directory_path or stored_pattern != pattern:
                self.index = 0
                self.WDB.insert('Batch Counters', label, 0)
                self.WDB.insert('Batch Paths', label, directory_path)
                self.WDB.insert('Batch Patterns', label, pattern)

            else:
                self.index = self.WDB.get('Batch Counters', label)

            self.label = label

        def load_images(self, directory_path, pattern):

            # 加载指定路径下的图像文件
            for file_name in glob.glob(os.path.join(glob.escape(directory_path), pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.image_paths.append(abs_file_path)

        def get_image_by_number(self, image_number):
            isTrue = 2
            for file_name in self.image_paths:
                single_image_path = file_name
                file_name_only = os.path.basename(file_name)
                # 提取图像名称中第一个逗号前的字符串
                file_number = file_name_only.split(',')[0]
                # 提取数字部分
                file_number = ''.join(filter(str.isdigit, file_number))
                if file_number == "":
                    continue
                if int(image_number) == int(file_number):
                    i = Image.open(file_name)
                    i = ImageOps.exif_transpose(i)
                    isTrue = 1
                    return (i, os.path.basename(file_name),single_image_path,isTrue)
            return (self.create_black_image(), f"编号{image_number}对应图像不存在，输出512*512黑色图像" , None , isTrue,)

        def get_image_by_id(self, image_id):

            # 根据图像ID获取图像和文件名
            if image_id < 0 or image_id >= len(self.image_paths):
                print(f"Invalid image index `{image_id}`")
                return
            i = Image.open(self.image_paths[image_id])
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(self.image_paths[image_id]))

        def get_next_image(self):

            # 获取下一张图像
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            self.index += 1
            if self.index == len(self.image_paths):
                self.index = 0
            print(f'{self.label} Index: {self.index}')
            self.WDB.insert('Batch Counters', self.label, self.index)
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(image_path))

        def get_current_image(self):

            # 获取当前图像的文件名
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            return os.path.basename(image_path)

        def create_black_image(self):
            # Creates a 512x512 black image
            return Image.fromarray(np.zeros((512, 512), dtype=np.uint8))

        def rename_images_with_sequence(self, folder_path):

            # 获取文件夹下所有文件
            files = os.listdir(folder_path)
            # 检查文件是否为图片文件
            def is_valid_image(file):
                return file.lower().endswith(ALLOWED_EXT)
            # 获取第一张图片文件
            first_image = next((file for file in files if is_valid_image(file)), None)
            # 如果没有图片文件，则直接返回
            if not first_image:
                print("没有图片文件")
                return

            # 检查所有图片文件的前缀名是否为纯数字
            all_prefixes_are_digits = all(os.path.splitext(file)[0].isdigit() for file in files if is_valid_image(file))
            if all_prefixes_are_digits:
                print("所有图片文件的前缀名都为纯数字，放弃重命名")
                return

            # 重命名图片文件
            for i, file in enumerate(files):
                if is_valid_image(file):
                    ext = os.path.splitext(file)[1]
                    new_name = f"{i:03d}{ext}"
                    old_path = os.path.join(folder_path, file)
                    new_path = os.path.join(folder_path, new_name)
                    os.rename(old_path, new_path)
            print("图片文件已成功重命名")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 打印文件名和行号

        if kwargs['mode'] != 'single_image':
            return float("NaN")
        else:
            fl = SAMIN_Load_Image_Batch.BatchImageLoader(kwargs['path'], kwargs['label'], kwargs['pattern'])
            filename = fl.get_current_image()
            image = os.path.join(kwargs['path'], filename)
            sha = get_sha256(image)
            return sha

#(pil2tensor(Image.new('RGB', (512,512), (0, 0, 0, 0))), 'null')

class SANMI_CounterNode:
    def __init__(self):
        self.HDB = WASDatabase(WAS_HISTORY_DATABASE)
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_number": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "end_number": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "label": ("STRING", {"default": 'Number 001', "multiline": False}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("number", "now_number_text")
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    def gogo(self, start_number, end_number,label='Number 001',index=0):
        # 获取数字列表
        number_list = list(range(start_number, end_number + 1))
        # 创建NumberListLoader对象
        Nu = self.NumberListLoader(number_list,label)
        # 获取当前数字,更新索引,并添加到WDB数据库中
        number = Nu.get_next_number()
        # 获取文本，提示当前数字
        now_number_text = f"现在输出的编号是：{number}"
        return (number, now_number_text)

    class NumberListLoader:
        def __init__(self, number_list,label):
            self.WDB = WDB
            # 初始化BatchImageLoader对象
            self.number_list = number_list
            stored_number_list = self.WDB.get('List', label)
            # 如果列表不匹配，则重置列表，并获取当前索引值
            if stored_number_list != number_list:
                self.index = 0
                self.WDB.insert('Counters', label, 0)
                self.WDB.insert('List', label, number_list)
            else:
                self.index = self.WDB.get('Counters', label)
            self.label = label


        def get_next_number(self):
            # 获取下一个数字
            if self.index >= len(self.number_list):
                self.index = 0
            number = self.number_list[self.index]
            self.index += 1
            if self.index == len(self.number_list):
                self.index = 0
            self.WDB.insert('Counters', self.label, self.index)
            return number
        def get_current_number(self):

            # 获取当前图像的文件名
            if self.index >= len(self.number_list):
                self.index = 0
            number = self.number_list[self.index]
            return number
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 打印文件名和行号
            Nu = SANMI_CounterNode.gogo(kwargs['start_number'], kwargs['label'], kwargs['end_number'])
            number = Nu.get_current_number()
            now_number_text = f"现在输出的数字是：{number}"
            return (number, now_number_text)


class SAMIN_Read_Prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "verbose": (["true", "false"],),
                "image_path": ("STRING", {"default": '', "multiline": False}),

            },
        }

    CATEGORY = "Sanmi Simple Nodes/Simple NODE"
    ''' Return order:
        positive prompt(string), negative prompt(string), seed(int), steps(int), cfg(float), 
        width(int), height(int)
    '''
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("image", "positive", "negative", "seed", "steps", "cfg", "width", "height")
    FUNCTION = "get_image_data"

    def get_image_data(self, image_path, verbose):
        with open(image_path, 'rb') as file:
            img = Image.open(file)
            if verbose:
                print(f"Loaded image with shape: {img.size}")
            extension = image_path.split('.')[-1]
            # Convert the image to RGB
            image = img.convert("RGB")
            # Normalize the image
            image = np.array(image).astype(np.float32) / 255.0
            # Convert the image to PyTorch tensor
            image = torch.from_numpy(image).permute(2, 0, 1)  # Change the dimensions to (C, H, W)
            # Add an extra dimension for batch size
            image = image.unsqueeze(0)

        parameters = ""
        comfy = False
        if extension.lower() == 'png':
            try:
                parameters = img.info['parameters']
                if not parameters.startswith("Positive prompt"):
                    parameters = "Positive prompt: " + parameters
            except:
                parameters = ""
                print("Error loading prompt info from png")
                # return "Error loading prompt info."
        elif extension.lower() in ("jpg", "jpeg", "webp"):
            try:
                exif = piexif.load(img.info["exif"])
                parameters = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
                parameters = piexif.helper.UserComment.load(parameters)
                if not parameters.startswith("Positive prompt"):
                    parameters = "Positive prompt: " + parameters
            except:
                try:
                    parameters = str(img.info['comment'])
                    comfy = True
                    # legacy fixes
                    parameters = parameters.replace("Positive Prompt", "Positive prompt")
                    parameters = parameters.replace("Negative Prompt", "Negative prompt")
                    parameters = parameters.replace("Start at Step", "Start at step")
                    parameters = parameters.replace("End at Step", "End at step")
                    parameters = parameters.replace("Denoising Strength", "Denoising strength")
                except:
                    parameters = ""
                    print("Error loading prompt info from jpeg")
                    # return "Error loading prompt info."

        if (comfy and extension.lower() == 'jpeg'):
            parameters = parameters.replace('\\n', ' ')
        else:
            parameters = parameters.replace('\n', ' ')

        patterns = [
            "Positive prompt: ",
            "Negative prompt: ",
            "Steps: ",
            "Start at step: ",
            "End at step: ",
            "Sampler: ",
            "Scheduler: ",
            "CFG scale: ",
            "Seed: ",
            "Size: ",
            "Model: ",
            "Model hash: ",
            "Denoising strength: ",
            "Version: ",
            "ControlNet 0",
            "Controlnet 1",
            "Batch size: ",
            "Batch pos: ",
            "Hires upscale: ",
            "Hires steps: ",
            "Hires upscaler: ",
            "Template: ",
            "Negative Template: ",
        ]
        if (comfy and extension.lower() == 'jpeg'):
            parameters = parameters[2:]
            parameters = parameters[:-1]

        keys = re.findall("|".join(patterns), parameters)
        values = re.split("|".join(patterns), parameters)
        values = [x for x in values if x]
        results = {}
        result_string = ""
        for item in range(len(keys)):
            result_string += keys[item] + values[item].rstrip(', ')
            result_string += "\n"
            results[keys[item].replace(": ", "")] = values[item].rstrip(', ')

        if (verbose == "true"):
            print(result_string)

        try:
            positive = results['Positive prompt']
        except:
            positive = ""
        try:
            negative = results['Negative prompt']
        except:
            negative = ""
        try:
            seed = int(results['Seed'])
        except:
            seed = -1
        try:
            steps = int(results['Steps'])
        except:
            steps = 20
        try:
            cfg = float(results['CFG scale'])
        except:
            cfg = 8.0
        try:
            width, height = img.size
        except:
            width, height = 512, 512

        ''' Return order:
            positive prompt(string), negative prompt(string), seed(int), steps(int), cfg(float), 
            width(int), height(int)
        '''

        return (image, positive, negative, seed, steps, cfg, width, height)


class SAMIN_String_Attribute_Selector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_A": ("STRING", {"multiline": False, "default": "text", "forceInput": True}),
                "text_B": ("STRING", {"multiline": False, "default": "text", "forceInput": True}),
                "prompt_type": (["Random", "PriorityB"], {"default": "PriorityB"}),
            },
        }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("prompt","full_prompt","clear_text_A", "clear_text_B", "name_A", "name_B")
    FUNCTION = "doit"
    OUTPUT_NODE = False
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    def __init__(self):
        pass

    @staticmethod
    def get_attributes(text):
        pattern = r'\[(.*?):(.*?)\]'
        attributes = {}
        matches = re.findall(pattern, text)
        for match in matches:
            attribute_name = match[0]
            attribute_value = match[1]
            if attribute_name in attributes:
                attributes[attribute_name].append(attribute_value)
            else:
                attributes[attribute_name] = [attribute_value]
        return attributes

    @staticmethod
    def get_prompt(attributes_A, attributes_B, prompt_type):
        new_prompt = []
        for attribute_name in attributes_A.keys():
            attribute_values_A = attributes_A[attribute_name]
            attribute_values_B = attributes_B.get(attribute_name, attribute_values_A)

            if prompt_type == "Random":
                new_prompt.append(random.choice(attribute_values_A + attribute_values_B))
            elif prompt_type == "PriorityB":
                new_prompt.append(attribute_values_B[-1])

        return new_prompt

    @staticmethod
    def remove_attributes(text):
        pattern = r"\[.*?\]"
        return re.sub(pattern, "", text)

    @staticmethod
    def get_name(text):
        start_index = text.find("<lora:") + len("<lora:")
        end_index = text.find(":", start_index)
        return text[start_index:end_index]

    def doit(self, text_A, text_B, prompt_type):

        attribute_values_A = self.get_attributes(text_A)
        attribute_values_B = self.get_attributes(text_B)
        new_prompt = self.get_prompt(attribute_values_A, attribute_values_B, prompt_type)
        clear_text_A = self.remove_attributes(text_A)
        clear_text_B = self.remove_attributes(text_B)
        prompt = "(" + ','.join(new_prompt[::-1]) + ")"
        full_prompt = clear_text_A + ',' + clear_text_B + ',' + prompt
        name_A = self.get_name(clear_text_A)
        name_B = self.get_name(clear_text_B)
        return prompt, full_prompt, clear_text_A, clear_text_B, name_A, name_B


class SANMIN_ClothingWildcards:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
        }

    CATEGORY = "Sanmi Simple Nodes/Simple NODE"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("full_prompt", "prompt", "cfg", "exposure", "skin",)
    FUNCTION = "doit"
    # 提取属性，并以字典形式返回
    def extract_attributes(self, populated_text):
        attributes = {}
        matches = re.findall(r"\[.*?\]", populated_text)
        for match in matches:
            match = match.rstrip(']')
            parts = re.findall(r"(\w+)\s*=\s*([^,]+)", match)
            for key, value in parts:
                attributes[key] = value
        return attributes
    # 查找输入的通配符文本，并返回文本列表。
    def find_matching_files(self, wildcard_text):
        wildcard_names = re.findall(r"__(.*?)__", wildcard_text)
        matching_files = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for wildcard_name in wildcard_names:
            file_path = os.path.join(script_dir, "wildcards", f"{wildcard_name}.txt")
            if os.path.exists(file_path):
                matching_files.append(file_path)
            else:
                matching_files.append(f"未查到该路径: {file_path}")
        return matching_files
    # 查找通配符为对应的文本，并返回随机一行。
    def replace_wildcards(self, wildcard_text):
        matching_files = self.find_matching_files(wildcard_text)
        for file_path in matching_files:
            if file_path.startswith("未查到该路径:"):
                continue
            file_content = self.get_file_content(file_path)
            wildcard_name = os.path.basename(file_path).split(".")[0]
            wildcard_lines = file_content.splitlines()
            wildcard_text = re.sub(re.escape(f"__{wildcard_name}__"), random.choice(wildcard_lines), wildcard_text)
        return wildcard_text
    # 读取通配符文本内容，并返回字符串。
    def get_file_content(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return file_content.strip()

    def doit(self, wildcard_text):
        matching_files = self.find_matching_files(wildcard_text)
        for file_path in matching_files:
            print("ClothingWildcards查找到的文件路径：",file_path)

        populated_text = self.replace_wildcards(wildcard_text)
        attributes = self.extract_attributes(populated_text)

        cfg = float(attributes.get("cfg", 7))
        exposure = float(attributes.get("exposure", 0))
        skin = float(attributes.get("skin", 0))

        # 去除属性部分，只返回正文部分
        prompt = re.sub(r"\[.*?\]", "", populated_text)
        return (populated_text, prompt.strip(), cfg, exposure, skin)


class SANMIN_EditWildcards:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wildcard_name": ("STRING", {"multiline": False, "dynamicPrompts": True}),
                "replace": ("BOOLEAN", {"default": False}),
                "replace_text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    CATEGORY = "Sanmi Simple Nodes/Simple NODE"
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("wildcard_name", )
    FUNCTION = "doit"

    # 查找输入的通配符文本，并返回文本列表。
    def find_matching_files(self, wildcard_name,replace_text):
        wildcard_names = re.findall(r"__(.*?)__", wildcard_name)
        matching_files = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for wildcard_name in wildcard_names:
            file_path = os.path.join(script_dir, "wildcards", f"{wildcard_name}.txt")
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(replace_text)  # 创建空文件
            matching_files.append(file_path)
        return matching_files

    def doit(self, wildcard_name, replace_text, replace):
        matching_files = self.find_matching_files(wildcard_name, replace_text)
        if replace:
            for file_path in matching_files:
                if os.path.exists(file_path) and replace_text:
                    with open(file_path, 'w') as f:
                        f.write(replace_text)
                    print("查找到的需要替换的文件：", file_path)
                    print("替换文本为：", replace_text)
                else:
                    print("未找到文件：", file_path)
        return (wildcard_name,)

class SANMIN_SimpleWildcards:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
        }

    CATEGORY = "Sanmi Simple Nodes/Simple NODE"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "doit"

    # 查找输入的通配符文本，并返回文本列表。
    def find_matching_files(self, wildcard_text):
        wildcard_names = re.findall(r"__(.*?)__", wildcard_text)
        matching_files = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for wildcard_name in wildcard_names:
            file_path = os.path.join(script_dir, "wildcards", f"{wildcard_name}.txt")
            if os.path.exists(file_path):
                matching_files.append(file_path)
            else:
                matching_files.append(f"未查到该路径: {file_path}")
        return matching_files
    # 查找通配符为对应的文本，并返回随机一行。
    def replace_wildcards(self, wildcard_text):
        matching_files = self.find_matching_files(wildcard_text)
        for file_path in matching_files:
            if file_path.startswith("未查到该路径:"):
                continue
            file_content = self.get_file_content(file_path)
            wildcard_name = os.path.basename(file_path).split(".")[0]
            wildcard_lines = file_content.splitlines()
            wildcard_text = re.sub(re.escape(f"__{wildcard_name}__"), random.choice(wildcard_lines), wildcard_text)
        return wildcard_text
    # 读取通配符文本内容，并返回字符串。
    def get_file_content(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return file_content.strip()

    def doit(self, wildcard_text):
        matching_files = self.find_matching_files(wildcard_text)
        for file_path in matching_files:
            print("SimpleWildcards查找到的文件路径：",file_path)
        prompt = self.replace_wildcards(wildcard_text)
        # 去除属性部分，只返回正文部分
        return (prompt.strip(),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        Nu = SANMIN_SimpleWildcards.doit(kwargs['wildcard_text'])
        prompt = Nu.replace_wildcards()

        return (prompt,)

class SimpleBatchImageLoader:
  def __init__(self, path, pattern='*'):
    self.path = path
    self.pattern = pattern

  def get_images(self):
    images = []
    # Iterate over files in the specified path
    for file in os.listdir(self.path):
      # Check if the file matches the pattern
      if file.endswith(self.pattern):
        # Add the file to the list of images
        images.append(os.path.join(self.path, file))
    return images


class LoadPathImagesPreview(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '*', "multiline": False}),
            },
        }

    NAME = "Images_Preview"
    FUNCTION = "preview_images"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    class BatchImageLoader:
        def __init__(self, directory_path, pattern):
            self.image_paths = []
            self.load_images(directory_path, pattern)
            self.image_paths.sort()

        def load_images(self, directory_path, pattern):
            for file_name in glob.glob(os.path.join(glob.escape(directory_path), pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.image_paths.append(abs_file_path)

    def get_image_by_path(self, image_path):
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        return image

    def preview_images(self, path, pattern='*', filename_prefix="sanmin.preview.", prompt=None, extra_pnginfo=None):
        fl = self.BatchImageLoader(path, pattern)
        images = []
        for image_path in fl.image_paths:
            image = Image.open(image_path)
            tensor_image = pil2tensor(image)
            image = tensor_image[0]
            images.append(image)
        if not images:
            raise ValueError("No images found in the specified path")

        return self.save_images(images, filename_prefix, prompt, extra_pnginfo)

def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

class SanmiSaveImageToLocal:

  def __init__(self):
    self.output_dir = folder_paths.get_output_directory()
    self.type = "output"
    self.prefix_append = ""
    self.compress_level = 4

  @classmethod
  def INPUT_TYPES(s):
    return {"required":
              {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "ComfyUI"}),
                "file_path": ("STRING", {"multiline": False, "default": "", "dynamicPrompts": False}),
                "isTrue": ("INT", {"default": 0}),
                "generate_txt": ("BOOLEAN", {"default": False}),
                "txt_content": ("STRING", {"multiline": True, "default": "", "hidden": True})
              },
              "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

  RETURN_TYPES = ()
  FUNCTION = "save"
  OUTPUT_NODE = True
  NAME = "Save images"
  CATEGORY = "Sanmi Simple Nodes/Simple NODE"

  def save(self, images, file_path , isTrue=0, filename="ComfyUI", prompt=None, extra_pnginfo=None, generate_txt=False, txt_content=""):
    if isTrue == 2:
        return ()
    if file_path == "":
        SaveImage().save_images(images, filename, prompt, extra_pnginfo)
        return ()

    if not os.path.exists(file_path):
        # 使用os.makedirs函数创建新目录
        os.makedirs(file_path)
        print("目录已创建")
    else:
        print("目录已存在")

    for image in images:
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        file = f"{filename}.png"
        fp = os.path.join(file_path, file)
        if os.path.exists(fp):
            file = f"{filename},{generate_random_string(8)}.png"
        img.save(os.path.join(file_path, file))

        if generate_txt:
            txt_filename = os.path.splitext(file)[0] + ".txt"
            txt_filepath = os.path.join(file_path, txt_filename)
            with open(txt_filepath, "w") as f:
                f.write(txt_content)
    return ()

class SANMI_ConvertToEnglish:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number": ("STRING", {"multiline": False, "default": "",}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    @staticmethod
    def gogo(number):
        mapping = {str(i): letter for i, letter in enumerate(string.ascii_uppercase)}
        output_string = ""
        for char in number:
            if char.isdigit():
                if char in mapping:
                    output_string += mapping[char]
                else:
                    output_string += char
            else:
                output_string += char
        return (output_string,)

class SANMI_ChineseToCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chinese_name": ("STRING", {"multiline": False, "default": "填写角色中文名",}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "find_character"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    @staticmethod
    # 读取角色英文名
    def find_character(chinese_name):
        # 获取当前代码文件的目录路径
        code_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建文件的完整路径
        file_path = os.path.join(code_dir, "animagine-xl-3.1-characterfull-zh.txt")

        # Read the file
        df = pd.read_csv(file_path, sep=",|#", engine='python', header=None)

        # Rename columns
        df.columns = ["Gender", "Character_EN", "Anime_EN", "Character_CN", "Anime_CN"]

        # Find character
        result = df[df['Character_CN'].str.contains(chinese_name)]

        # If result is found, return the corresponding information
        if not result.empty:
            gender = result['Gender'].values[0]
            character_en = result['Character_EN'].values[0]
            anime_en = result['Anime_EN'].values[0]
            character = f"{gender}, {character_en}, {anime_en}"

            return (character,)
        else:
            return ("Character not found",)


class SANMIN_AdjustTransparency:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert": ("BOOLEAN", {"default": False}),
                "complete_elimination": ("BOOLEAN", {"default": False}),
                "opacity": ("FLOAT", {"default": 100, "min": 0, "max": 100, "step": 5}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_transparency"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"
    def adjust_transparency(self, image, mask, opacity, invert=False, complete_elimination=False):

        # Convert the image tensor to a PIL Image
        image = tensor2pil(image)
        # Convert the mask tensor to a PIL Image
        mask = tensor2pil(mask)

        # Convert the mask to grayscale
        mask = mask.convert('L')

        # Invert the mask if required
        if invert:
            mask = ImageOps.invert(mask)

        # Apply the mask to the image
        if complete_elimination:
            # Create a new image with the same size as the original image
            new_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
            # Paste the original image onto the new image, but only where the mask is not zero
            new_image.paste(image, mask=mask)
            image = new_image
        else:
            image.putalpha(mask)

        # Adjust the opacity
        r, g, b, a = image.split()
        a = a.point(lambda x: int(x * (opacity / 100)))
        image.putalpha(a)

        return (pil2tensor(image),)


class SANMIN_Adapt_Coordinates:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates": ("STRING", {"multiline": False, "forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coord_str",)
    FUNCTION = "adapt_coordinates"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    def adapt_coordinates(self, coordinates, width, height):
        scale_factor_x = round(width / 512)
        scale_factor_y = round(height / 512)

        coordinates_out = [
            {
                "x": coord["x"] * scale_factor_x if coord["x"] != 0 else 0,
                "y": coord["y"] * scale_factor_y if coord["y"] != 0 else 0,
            }
            for coord in eval(coordinates)
        ]

        coordinates_out = str(coordinates_out)
        print("coordinates_out1:", type(coordinates_out), coordinates_out)
        return (coordinates_out,)


class SCALE_AND_FILL_BLACK:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "percentage": ("FLOAT", {"default": 100, "min": 1, "max": 100, "step": 5}),
                "image": ("IMAGE", ),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "scale_and_fill_black"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    def scale_and_fill_black(self, percentage, image):

        # Convert tensor to PIL image
        pil_image = tensor2pil(image)

        # Calculate the new size of the image based on the percentage
        width, height = pil_image.size
        new_width = int(width * (percentage / 100.0))
        new_height = int(height * (percentage / 100.0))

        # Resize the image using nearest-exact interpolation
        pil_image = pil_image.resize((new_width, new_height), Image.NEAREST)

        # Create a new image with the original size and fill it with black background
        new_pil_image = Image.new('RGB', (width, height), (0, 0, 0))

        # Paste the resized image onto the new image at the center
        x = (width - new_width) // 2
        y = (height - new_height) // 2
        new_pil_image.paste(pil_image, (x, y))

        # Convert the PIL image back to tensor
        image = pil2tensor(new_pil_image)
        return (image,)


class Upscale_And_Original_Size:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scale_by": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 9.99, "step": 0.01}),
                "image": ("IMAGE", ),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_and_original_size"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    def upscale_and_original_size(self, scale_by, image):
        # Convert input image to PIL format
        image_pil = tensor2pil(image)

        # Calculate new size after upscaling
        original_size = image_pil.size
        new_width = int(original_size[0] * scale_by)
        new_height = int(original_size[1] * scale_by)

        # Resize the image using nearest neighbor interpolation
        resized_image = image_pil.resize((new_width, new_height), resample=Image.NEAREST)

        # Calculate crop box coordinates for center cropping
        left = (new_width - original_size[0]) // 2
        top = (new_height - original_size[1]) // 2
        right = left + original_size[0]
        bottom = top + original_size[1]

        # Perform center crop on the resized image
        cropped_image = resized_image.crop((left, top, right, bottom))

        # Convert cropped image back to tensor format
        cropped_image_tensor = pil2tensor(cropped_image)

        return (cropped_image_tensor,)


def gaussian_kernel(kernel_size: int, sigma: float):
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size), torch.linspace(-1, 1, kernel_size), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()

class SANMIN_BlurMaskArea:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "blur_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blur_mask_area"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    def blur_mask_area(self, image: torch.Tensor, mask: torch.Tensor, blur_radius: int, sigma: float):
        if blur_radius == 0:
            return (image,)

        batch_size, height, width, channels = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma).repeat(channels, 1, 1).unsqueeze(1)

        image = image.permute(0, 3, 1, 2)  # Torch wants (B, C, H, W) we use (B, H, W, C)
        padded_image = F.pad(image, (blur_radius, blur_radius, blur_radius, blur_radius), 'reflect')
        blurred = F.conv2d(padded_image, kernel, padding=kernel_size // 2, groups=channels)
        blurred = blurred[:, :, blur_radius:-blur_radius, blur_radius:-blur_radius]

        blurred = blurred.permute(0, 2, 3, 1)
        image = image.permute(0, 2, 3, 1)

        # Apply the mask to blend the blurred and original images
        mask = mask.unsqueeze(-1).expand_as(image)
        output_image = mask * blurred + (1 - mask) * image

        return (output_image,)

# 浮点数
class Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("FLOAT", {"default": 0, "step": 0.1})},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "execute"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    def execute(self, value):
        return (value,)


#定义功能和模块名称
# NOTE: names should be globally unique
# 不能有下划线在命名时
NODE_CLASS_MAPPINGS = {
    "Samin Load Image Batch": SAMIN_Load_Image_Batch,
    "Samin Counter": SANMI_CounterNode,
    "Image Load with Metadata ": SAMIN_Read_Prompt,
    "SAMIN String Attribute Selector": SAMIN_String_Attribute_Selector,
    "SANMIN ClothingWildcards":SANMIN_ClothingWildcards,
    "SANMIN SimpleWildcards":SANMIN_SimpleWildcards,
    "SANMIN LoadPathImagesPreview": LoadPathImagesPreview,
    "SANMIN SanmiSaveImageToLocal": SanmiSaveImageToLocal,
    "SANMIN ConvertToEnglish": SANMI_ConvertToEnglish,
    "SANMIN ChineseToCharacter": SANMI_ChineseToCharacter,
    "SANMIN AdjustTransparency": SANMIN_AdjustTransparency,
    "SANMIN Adapt Coordinates": SANMIN_Adapt_Coordinates,
    "SANMIN SCALE AND FILL BLACK": SCALE_AND_FILL_BLACK,
    "SANMIN Upscale And Original Size": Upscale_And_Original_Size,
    "SANMIN BlurMaskArea": SANMIN_BlurMaskArea,
    "SANMIN EditWildcards": SANMIN_EditWildcards,
    "SANMIN Float": Float,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Samin Counter": "Counter",
    "Image Load with Metadata ": "Read_Prompt",
    "SAMIN String Attribute Selector": "String_Attribute_Selector",
    "SANMIN ClothingWildcards":"ClothingWildcards",
    "SANMIN SimpleWildcards":"SimpleWildcards",
    "SANMIN LoadPathImagesPreview": "LoadPathImagesPreview",
    "SANMIN SanmiSaveImageToLocal": "SanmiSaveImageToLocal",
    "SANMIN ConvertToEnglish": "ConvertToEnglish",
    "SANMIN ChineseToCharacter": "ChineseToCharacter",
    "SANMIN CropByMask": "CropByMask",
    "SANMIN AdjustTransparency": "AdjustTransparency",
    "SANMIN Adapt Coordinates": "Adapt_Coordinates",
    "SANMIN SCALE AND FILL BLACK": "scale_and_fill_black",
    "SANMIN Upscale And Original Size": "Upscale_And_Original_Size",
    "SANMIN BlurMaskArea": "sanmi_BlurMaskArea",
    "SANMIN EditWildcards": "sanmi_EditWildcards",
    "SANMIN Float": "sanmi_Float",
}




