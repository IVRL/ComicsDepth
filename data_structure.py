#!/usr/bin/env python
# coding: utf-8

"""
data_structure.py: Main utilities about how to handle the data structure
"""


from typing import List


########
# Imports
########

import os.path
import warnings


########
# Main data folder
########

DATA_FOLDER = os.path.normpath("data")


########
# DCM dataset
########

DCM_FOLDER = os.path.join(DATA_FOLDER, "dcm_dataset.git")
DCM_IMAGES_FOLDER = os.path.join(DCM_FOLDER, "images")
DCM_ANNOTATIONS_FOLDER = os.path.join(DCM_FOLDER, "groundtruth")
DCM_TRAIN_FILE = os.path.join(DCM_FOLDER, "train.txt")
DCM_VALIDATION_FILE = os.path.join(DCM_FOLDER, "val.txt")
DCM_TEST_FILE = os.path.join(DCM_FOLDER, "test.txt")
DCM_FILENAMES = {
    "folder": DCM_FOLDER,
    "images": DCM_IMAGES_FOLDER,
    "annotations": DCM_ANNOTATIONS_FOLDER,
    "train": DCM_TRAIN_FILE,
    "validation": DCM_VALIDATION_FILE,
    "test": DCM_TEST_FILE
}


def ASSERT_DCM() -> None:
    """Asserts that the DCM dataset structure is fine.
    """
    for folder in ["folder", "images", "annotations"]:
        assert os.path.exists(DCM_FILENAMES[folder])
        assert os.path.isdir(DCM_FILENAMES[folder])
    for file in ["train", "validation", "test"]:
        print(DCM_FILENAMES[file])
        assert os.path.exists(DCM_FILENAMES[file])
        assert os.path.isfile(DCM_FILENAMES[file])


def DCM_GET_FILES_LIST(subset_filename: str) -> List[str]:
    """Gets the files list of the given subset.

    Args:
        subset_filename (str): The desired subset.

    Returns:
        List[str]: The files list of the given subset.
    """
    with open(subset_filename) as subset_file:
        lines = subset_file.readlines()
    files_list = [line.rstrip() for line in lines if line.rstrip() != ""]
    return files_list


def DCM_IMAGE_PATH_FROM_NAME(filename: str) -> str:
    """Gets the image path from the given image name.

    Args:
        filename (str): The name of the desired image.

    Returns:
        str: The image path of the desired image.
    """
    image_path = os.path.join(
        DCM_FILENAMES["images"],
        filename+".jpg"
    )
    return image_path


def DCM_ANNOTATIONS_PATH_FROM_NAME(filename: str) -> str:
    """Gets the annotations path from the given image name.

    Args:
        filename (str): The name of the desired image.

    Returns:
        str: The annotations path of the desired image.
    """
    annot_path = os.path.join(
        DCM_FILENAMES["annotations"],
        filename+".txt"
    )
    return annot_path


def DCM_READ_ANNOTATIONS(annot_path: str) -> List[List[int]]:
    """Gets the annotations from the annotations file path.

    Args:
        annot_path (str): The desired annotations file path.

    Returns:
        List[List[int]]: The annotations
            (class_id, x1_frame, y1_frame, x2_frame, y2_frame)
            from the annotations file path in a list.
    """
    with open(annot_path) as annot:
        annotations = annot.readlines()
    annotations_list_str = [annotation.split(
        " ") for annotation in annotations]
    annotations_list_int = []
    for annotation in annotations_list_str:
        class_id, x1_frame, y1_frame, x2_frame, y2_frame = annotation
        annotations_list_int.append([
            int(class_id),
            int(x1_frame),
            int(y1_frame),
            int(x2_frame),
            int(y2_frame)
        ])
    return annotations_list_int


########
# DCM cropped frames
########
DCM_CROPPED_FOLDER = os.path.join(DATA_FOLDER, "dcm_cropped")
DCM_CROPPED_IMAGES_FOLDER = os.path.join(DCM_CROPPED_FOLDER, "images")
DCM_CROPPED_TRAIN_FILE = os.path.join(DCM_CROPPED_FOLDER, "train.txt")
DCM_CROPPED_VALIDATION_FILE = os.path.join(
    DCM_CROPPED_FOLDER, "validation.txt")
DCM_CROPPED_TEST_FILE = os.path.join(DCM_CROPPED_FOLDER, "test.txt")
DCM_CROPPED_FILENAMES = {
    "folder": DCM_CROPPED_FOLDER,
    "images": DCM_CROPPED_IMAGES_FOLDER,
    "train": DCM_CROPPED_TRAIN_FILE,
    "validation": DCM_CROPPED_VALIDATION_FILE,
    "test": DCM_CROPPED_TEST_FILE
}


def ASSERT_DCM_CROPPED() -> None:
    """ Asserts that the "cropped DCM" dataset structure is fine.
    """
    for folder in ["folder", "images"]:
        assert os.path.exists(DCM_CROPPED_FILENAMES[folder])
        assert os.path.isdir(DCM_CROPPED_FILENAMES[folder])
    for file in ["train", "validation", "test"]:
        assert os.path.exists(DCM_CROPPED_FILENAMES[file])
        assert os.path.isfile(DCM_CROPPED_FILENAMES[file])


def CREATE_DCM_CROPPED() -> None:
    """ Creates the "cropped DCM" folder.
    """
    if os.path.exists(DCM_CROPPED_FILENAMES["folder"]):
        assert os.path.isdir(DCM_CROPPED_FILENAMES["folder"])
        warnings.warn("The \"cropped DCM\" folder already exists")
    os.makedirs(DCM_CROPPED_FILENAMES["folder"], exist_ok=True)


def DCM_CROPPED_IMAGE_PATH_FROM_NAME(filename: str) -> str:
    """Gets the image path from the given image name.

    Args:
        filename (str): The name of the desired image.

    Returns:
        str: The image path of the desired image.
    """
    image_path = os.path.join(
        DCM_CROPPED_FILENAMES["images"],
        filename+".jpg"
    )
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    return image_path
