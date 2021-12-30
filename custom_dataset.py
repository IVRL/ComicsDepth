#!/usr/bin/env python
# coding: utf-8

"""
custom_dataset.py: Code for the custom Dataset
"""

import numpy as np
import cv2

from typing import List, Tuple

# Partially based on https://github.com/eriklindernoren/PyTorch-GAN/blob/80e7702c25266925774d020e047fdff8d44f7a74/implementations/cyclegan/datasets.py
# Partially based on https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py
# Partially based on https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/utils.py
# Partially based on https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image
# Partially based on https://pytorch.org/hub/intelisl_midas_v2/

from utils.data_structure import *
import warnings

import glob
import random
import os
import torch.utils.data
import PIL
import torchvision.transforms as transforms


class CustomDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset to train our models
    """

    def __init__(self,
                 dcm: List[str] = ["train", "validation", "test"],
                 coco17: bool = True,
                 eBDtheque: bool = False,
                 eBDtheque_cropped: bool = False,
                 eBDtheque_cropped_mask: bool = False,
                 natural_depth: bool = True,
                 unaligned: bool = False,
                 resize: Tuple[int, int] = (384, 384),
                 interpolation: int = PIL.Image.BICUBIC,
                 resize_mode: str = "train",
                 max_len: int = 0,
                 ):
        """Init.

        Args:
            dcm (List[str], optional):
                Subsets of the DCM dataset to use.
                Defaults to ["train", "validation", "test"].
            coco17 (bool, optional):
                True: Use the COCO 2017 validation dataset.
                False: Do not use.
                Defaults to True.
            eBDtheque (bool, optional):
                True: Use the eBDtheque dataset.
                False: Do not use.
                Defaults to False.
            eBDtheque_cropped (bool, optional): 
                True: Use the "cropped" eBDtheque dataset.
                False: Do not use.
                Defaults to False.
            eBDtheque_cropped_mask (bool, optional):
                True: Use the "cropped" eBDtheque dataset with text masks.
                False: Do not use.
                Defaults to False.
            coco17_depth (bool, optional): 
                True: Use the COCO 2017 validation dataset with MiDas depth.
                False: Do not use.
                Defaults to True.
            unaligned (bool, optional):
                Whether to unalign the "natural" and the "comics" domains.
                True: Different indexes are used for both domains.
                False: Same index is used for both domains.
                Defaults to True.
            resize (Tuple[int, int], optional):
                Desired size.
                Defaults to (384, 384).
            interpolation (int, optional):
                Interpolation method.
                Defaults to PIL.Image.BICUBIC.
            resize_mode (str, optional): 
                "train": 'lower_bound' resize to at least the desired size,
                    and then random crop to exactely the desired size.
                "inference": 'upper_bound' resize to at most the desired size.
                Defaults to "train".
            max_len (int, optional): 
                0: No restriction on the length of the Dataset.
                max_len>=1: Restricts the length of the Dataset to max_len.
                Defaults to 0.
        """
        # Checking the given parameter
        use_dcm = len(dcm) > 0
        use_eBDtheque = eBDtheque or eBDtheque_cropped or eBDtheque_cropped_mask
        if use_dcm and use_eBDtheque:
            raise Exception("Please use only one comics dataset.")
        if eBDtheque_cropped_mask and not eBDtheque_cropped:
            warnings.warn(
                "eBDtheque_cropped is False but eBDtheque_cropped_mask is True.")
        if eBDtheque and (eBDtheque_cropped or eBDtheque_cropped_mask):
            warnings.warn(
                "eBDtheque (not cropped) and (eBDtheque_cropped or eBDtheque_cropped_mask) are both True.")
        if resize != (384, 384):
            warnings.warn("resize != (384, 384).")
        if resize_mode not in ["train", "inference"]:
            raise Exception("resize_mode not in [\"train\", \"inference\"].")
        if max_len >= 1:
            warnings.warn(
                "Length of the Dataset is restricted to "+str(max_len)+".")

        self.list_comics = []
        for subset in dcm:
            self.list_comics += [DCM_IMAGE_PATH_FROM_NAME(
                x) for x in DCM_GET_FILES_LIST(DCM_FILENAMES[subset])]

        self.list_natural = []
        if coco17:
            self.list_natural = sorted(glob.glob("data/coco_val2017/*.*"))
            self.list_natural = [x for x in self.list_natural if x[-3:]!=".md"]

        self.resize_mode = resize_mode

        resize_method = "lower_bound" if resize_mode == "train" else "upper_bound"
        self.resize = Resize(
            resize[0],
            resize[1],
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method=resize_method,
            image_interpolation_method=interpolation
        )

        self.unaligned = unaligned
        
        self.prepare = PrepareForNet()
        
        self.max_len = max_len


    def __getitem__(self, index):
        index2 = random.randrange(0, len(self.list_comics)) if self.unaligned else index
        
        
        #https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/utils.py#L98
               
        img_coco17 = cv2.imread(self.list_natural[index % len(self.list_natural)])
        if img_coco17.ndim == 2:
            img_coco17 = cv2.cvtColor(img_coco17, cv2.COLOR_GRAY2BGR)
        img_coco17 = cv2.cvtColor(img_coco17, cv2.COLOR_BGR2RGB) / 255.0
            
        size = img_coco17.shape
        
        img_coco17 = self.resize({"image": img_coco17})["image"]
        img_coco17 = self.prepare({"image": img_coco17})["image"]
        
        return {"img_coco17":img_coco17,
                "name_coco17":self.list_natural[index % len(self.list_natural)],
                "size_coco17":size
               }    
        
    def __len__(self):
        length = max(len(self.list_comics), len(self.list_natural))
        if self.max_len>=1:
            length = min(length, self.max_len)
        return length
        


def to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    """Converts the image to RGB mode.

    Args:
        image (PIL.Image.Image): Input image

    Returns:
        PIL.Image.Image: Output image in RGB mode
    """
    rgb_image = PIL.Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def to_l(image: PIL.Image.Image) -> PIL.Image.Image:
    """Converts the image to L mode.

    Args:
        image (PIL.Image.Image): Input image

    Returns:
        PIL.Image.Image: Output image in L mode
    """
    l_image = PIL.Image.new("L", image.size)
    l_image.paste(image)
    return l_image


def get_random_crop(image, image2, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = 0 if max_x == 0 else np.random.randint(0, max_x)
    y = 0 if max_y == 0 else np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    if image2 is not None:
        crop2 = image2[y: y + crop_height, x: x + crop_width]
    else:
        crop2 = None
    return crop, crop2

# Based on https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py

class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample
    
class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width: int,
        height: int,
        resize_target: bool = True,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        resize_method: str = "lower_bound",
        image_interpolation_method: int = cv2.INTER_AREA,
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=1, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width,
                                      height), interpolation=cv2.INTER_NEAREST
                )

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["mask"] = sample["mask"].astype(bool)

        return sample
