#!/usr/bin/env python
# coding: utf-8

"""
depth_estimator.py: Code about the depth estimation on top of the I2I translation
"""


from utils.data_structure import *
from utils.custom_dataset import CustomDataset

def optim_simple():
    import depth_simple_optim_lr
    depth_simple_optim_lr.main()
def optim_add_text():
    import depth_add_text_optim_lr
    depth_add_text_optim_lr.main()
    
def train_simple(lr):
    import depth_simple_train
    depth_simple_train.main(lr)
def train_add_text(lr):
    import depth_add_text_train
    depth_add_text_train.main(lr)
def train_add_text_ignoretext(lr):
    import depth_add_text_ignoretext_train
    depth_add_text_ignoretext_train.main(lr)
         
def optimize_lr(model):
    if model == "simple":
        optim_simple()
    elif model == "add_text":
        optim_add_text()
    else:
        raise NotImplementedError
        
def train(model, lr):
    if model == "simple":
        train_simple(lr)
    elif model == "add_text":
        train_add_text(lr)
    elif model == "add_text_ignoretext":
        train_add_text_ignoretext(lr)
    else:
        raise NotImplementedError
        
def generate_depth_images(model, lr):
    if model == "simple":
        import depth_simple_generatedeptheval
        depth_simple_generatedeptheval.main(lr)
    elif model == "add_text":
        import depth_add_text_generatedeptheval
        depth_add_text_generatedeptheval.main(lr)
    elif model == "add_text_ignoretext":
        import depth_add_text_ignoretext_generatedeptheval
        depth_add_text_ignoretext_generatedeptheval.main(lr)
    else:
        raise NotImplementedError
    
    
    
if __name__ == "__main__":
    
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        type=str,
                        help="which model to use (simple, add_text, add_text_ignoretext)")
    
    group = parser.add_mutually_exclusive_group()    
    group.add_argument(
        "--train",
        help="training the model",
        action="store_true"
    )
    group.add_argument(
        "--optimize_lr",
        help="optimizing the learning rate",
        action="store_true"
    )
    group.add_argument(
        "--generate_depth_images",
        help="generating depth images (trained depth estimator on comics)",
        action="store_true"
    )
        
    args = parser.parse_args()

    if args.optimize_lr:
        optimize_lr(args.model)
    if args.train:
        train(args.model, lr=1e-6)
    if args.generate_depth_images:
        generate_depth_images(args.model, lr=1e-6)
