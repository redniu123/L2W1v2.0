# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Modified for L2W1 Project - Minimal standalone version
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
L2W1 精简版工具函数

只包含图像加载相关的工具函数
"""

import logging
import os
import cv2
import numpy as np


def _check_image_file(path):
    """检查是否为图像文件"""
    img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file, infer_list=None):
    """
    获取图像文件列表
    
    Args:
        img_file: 图像文件或目录路径
        infer_list: 推理列表文件路径
        
    Returns:
        图像文件路径列表
    """
    imgs_lists = []
    if infer_list and not os.path.exists(infer_list):
        raise Exception("not found infer list {}".format(infer_list))
    if infer_list:
        with open(infer_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.strip().split("\t")[0]
            image_path = os.path.join(img_file, image_path)
            imgs_lists.append(image_path)
    else:
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))

        if os.path.isfile(img_file) and _check_image_file(img_file):
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            for single_file in os.listdir(img_file):
                file_path = os.path.join(img_file, single_file)
                if os.path.isfile(file_path) and _check_image_file(file_path):
                    imgs_lists.append(file_path)

    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def check_and_read(img_path):
    """
    检查并读取图像文件
    
    支持 GIF 和 PDF 格式
    
    Args:
        img_path: 图像文件路径
        
    Returns:
        (image, is_gif, is_pdf) 或 (images_list, False, True) for PDF
    """
    if os.path.basename(img_path)[-3:].lower() == "gif":
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            logger = logging.getLogger("ppocr")
            logger.info("Cannot read {}. This gif image maybe corrupted.".format(img_path))
            return None, False, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    elif os.path.basename(img_path)[-3:].lower() == "pdf":
        try:
            from paddle.utils import try_import
            fitz = try_import("fitz")
            from PIL import Image

            imgs = []
            with fitz.open(img_path) as pdf:
                for pg in range(0, pdf.page_count):
                    page = pdf[pg]
                    mat = fitz.Matrix(2, 2)
                    pm = page.get_pixmap(matrix=mat, alpha=False)

                    if pm.width > 2000 or pm.height > 2000:
                        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    imgs.append(img)
                return imgs, False, True
        except Exception as e:
            logger = logging.getLogger("ppocr")
            logger.warning(f"Cannot read PDF file {img_path}: {e}")
            return None, False, False
    return None, False, False


def binarize_img(img):
    """二值化图像"""
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img


def alpha_to_color(img, alpha_color=(255, 255, 255)):
    """将 RGBA 图像转换为 RGB"""
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img

