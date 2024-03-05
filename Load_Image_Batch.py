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



from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
from typing import Optional, Union, List
from urllib.request import urlopen
import comfy.diffusers_convert
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.clip_vision
import comfy.model_management
import folder_paths as comfy_paths
from comfy_extras.chainner_models import model_loading
import ast
import glob
import hashlib
import json
import nodes
import math
import numpy as np
from numba import jit
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
from tqdm import tqdm
import string
from PIL import Image,ImageOps
import glob
import json
import numpy as np
import os
import torch
import inspect



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
                "image_number": ("INT",{"default": 0, "min": 0, "max": 150000, "step": 1,"forceInput": False}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"],),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING")
    RETURN_NAMES = ("image","filename_text")
    FUNCTION = "load_batch_images"
    CATEGORY = "Sanmi Simple Nodes/Simple NODE"

    def load_batch_images(self, path, pattern='*', index=0, mode="single_image", label='Batch 001', allow_RGBA_output='false', filename_text_extension='true', image_number=None):

        allow_RGBA_output = (allow_RGBA_output == 'true')

        # 如果指定的路径不存在，则返回一个包含None的元组
        if not os.path.exists(path):
            return (None, )

        # 创建BatchImageLoader对象并获取图像路径
        fl = self.BatchImageLoader(path, label, pattern)
        # 所有符合规则的图像升序的绝对路径列表
        new_paths = fl.image_paths

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
                return (None, None)
        if mode == 'number_image':
            image, filename = fl.get_image_by_number(image_number)
            if image == None:
                print(f"No valid image was found for the next ID. Did you remove images from the source directory?")
                return (self.create_black_image(), None)
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
        return (pil2tensor(image), filename)


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
            for file_name in self.image_paths:
                file_name_only = os.path.basename(file_name)
                # 提取图像名称中第一个逗号前的字符串
                file_number = file_name_only.split(',')[0]
                # 提取数字部分
                file_number = ''.join(filter(str.isdigit, file_number))
                if int(image_number) == int(file_number):
                    i = Image.open(file_name)
                    i = ImageOps.exif_transpose(i)
                    return (i, os.path.basename(file_name))
            return self.create_black_image(), f"编号{image_number}对应图像不存在，输出512*512黑色图像"

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

#定义功能和模块名称
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Samin Load Image Batch": SAMIN_Load_Image_Batch,
    "Samin Counter": SANMI_CounterNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Samin Load Image Batch": "Samin Load Image Batch",
    "Samin Counter": "Samin Counter",
}




