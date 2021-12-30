#!/usr/bin/env python
# coding: utf-8

"""
 Code for I2I translation
"""


from utils.data_structure import *
from utils.custom_dataset import CustomDataset




def train_simple():
    import cyclegan_natural_comics_simple_train
    cyclegan_natural_comics_simple_train.main()
def train_depth_aware():
    import cyclegan_natural_comics_depth_aware_train
    cyclegan_natural_comics_depth_aware_train.main()
def train_without_text():
    import cyclegan_natural_comics_without_text_train
    cyclegan_natural_comics_without_text_train.main()
def train_add_text():
    import cyclegan_natural_comics_add_text_train
    cyclegan_natural_comics_add_text_train.main()
def train(model):
    if model == "simple":
        train_simple()
    elif model == "depth_aware":
        train_depth_aware()
    elif model == "without_text":
        train_without_text()
    elif model == "add_text":
        train_add_text()     
    else:
        raise NotImplementedError
         
def generate_depth_images(model):
    if model == "simple":
        import cyclegan_natural_comics_simple_generatedeptheval
        cyclegan_natural_comics_simple_generatedeptheval.main()
    elif model == "depth_aware":
        import cyclegan_natural_comics_depth_aware_generatedeptheval
        cyclegan_natural_comics_depth_aware_generatedeptheval.main()
    else:
        raise NotImplementedError
        



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        type=str,
                        help="which model to use (simple, depth_aware, without_text, add_text)")
    
    group = parser.add_mutually_exclusive_group()    
    group.add_argument(
        "--train",
        help="training the model",
        action="store_true"
    )
    group.add_argument(
        "--generate_depth_images",
        help="generating depth images (comics2natural2depth)",
        action="store_true"
    )
    
    args = parser.parse_args()

    if args.train:
        train(args.model)
    if args.generate_depth_images:
        generate_depth_images(args.model)
