#!/usr/bin/env python
# coding: utf-8

"""
evaluation.py: Code evaluate the results
"""
from utils.data_structure import *
from utils.custom_dataset import CustomDataset

def evalulate_image(prediction, gt_depth_ordering):
    with open(gt_depth_ordering) as file:
        lines = file.readlines()
    n = len(lines)
    goodl1 = 0
    alll1 = 0
    goodl2 = 0
    alll2 = 0
    for i1 in range(n):
        line1 = lines[i1].replace("\n","")
        l1, l2, x, y = line1.split(" ")
        l1, l2, x, y = int(l1), int(l2), int(x), int(y)
        for i2 in range(i1):
            line2 = lines[i2].replace("\n","")
            l1_, l2_, x_, y_ = line2.split(" ")
            l1_, l2_, x_, y_ = int(l1_), int(l2_), int(x_), int(y_)
            
            if l1<l1_:
                goodl1 += int(prediction[y][x] > prediction[y_][x_])
                alll1 += 1
            elif l1_<l1:
                goodl1 += int(prediction[y_][x_] > prediction[y][x])
                alll1 += 1
            else:
                if l2<l2_:
                    goodl2 += int(prediction[y][x] > prediction[y_][x_])
                    alll2 += 1
                elif l2_<l2:
                    goodl2 += int(prediction[y_][x_] > prediction[y][x])
                    alll2 += 1
    return goodl1, alll1, goodl2, alll2

def evaluate(approach, model):
    import glob
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    import utils.image_grid
    from torchvision.utils import save_image
    import torch
    import cv2
    import matplotlib.pyplot as plt
    import os.path
    import statistics
    
    if approach == "natural2comics":
        if model == "simple": to_evaluate = "approach1epoch99"
        elif model == "depth_aware": to_evaluate = "apporoach1v1epoch99"
        else: raise NotImplementedError
    elif approach == "depth_estimator":
        if model == "simple": to_evaluate = "approach21e-6epoch99"
        elif model == "add_text": to_evaluate = "approach2v11e-6epoch99"
        elif model == "add_text_ignoreloss": to_evaluate = "approach2v21e-6epoch99"
        else: raise NotImplementedError
    elif  approach == "baseline":
        if model == "no_batchnorm_trick": to_evaluate = "depth_no_batchnorm_trick"
        elif model == "batchnorm_trick": to_evaluate = "depth_batchnorm_trick"
        else: raise NotImplementedError
    else:
        raise NotImplementedError
    
    filename = "data/dcm_cropped/validation.txt"
    with open(filename) as file:
        names = file.read().split("\n")
        
    import math

    inter_model = []
    intra_model = []
    inter_midas = []
    intra_midas = []
    inter_relat = []
    intra_relat = []

    for name in names[:-1]:
        if not os.path.exists("data/dcm_cropped/depth/"+name+".txt"):
            print ("No GT depth ordering for", name)
        else:   
            print(name)
            filename = "data/dcm_cropped/images/"+name+".jpg"
            img = Image.open(filename)
            filename = "data/dcm_cropped/"+to_evaluate+"/"+name+"_originalsize.png"
            img_depth = Image.open(filename)
            filename = "data/dcm_cropped/depth_batchnorm_trick/"+name+"_originalsize.png"
            img_depth_midas = Image.open(filename)

            img = transforms.ToTensor()(img)
            img_depth = transforms.ToTensor()(img_depth)[0,:,:]
            img_depth_midas = transforms.ToTensor()(img_depth_midas)[0,:,:]

            gt_depth_ordering = "data/dcm_cropped/depth/"+name+".txt"
            goodl1, alll1, goodl2, alll2 = evalulate_image(img_depth, gt_depth_ordering)
            if alll1>0: 
                inter_model.append(goodl1/alll1) 
            if alll2>0: 
                intra_model.append(goodl2/alll2) 
            goodl1_, alll1_, goodl2_, alll2_ = evalulate_image(img_depth_midas, gt_depth_ordering)
            if alll1_>0: 
                inter_midas.append(goodl1_/alll1_) 
            if alll2_>0: 
                intra_midas.append(goodl2_/alll2_) 

            if goodl1_>0: 
                inter_relat.append(goodl1/goodl1_) 
            if goodl2_>0:
                intra_relat.append(goodl2/goodl2_) 


    inter_relat_log = [10*math.log10(x) for x in inter_relat]
    intra_relat_log = [10*math.log10(x) for x in intra_relat]


    def print_accuracy(listacc):
        print("-mean()", statistics.mean(listacc))
        print("-quantiles()", statistics.quantiles(listacc))
        print("-stdev()", statistics.stdev(listacc))
        print("-len()", len(listacc))
        print("-len>0()", sum([x>0 for x in listacc]))
        print("-len>1()", sum([x>1 for x in listacc]))
    print ("interobjects accuracy model")
    print_accuracy(inter_model)   
    print ("intraobjects accuracy model")  
    print_accuracy(intra_model)    
    print ("interobjects accuracy midas")
    print_accuracy(inter_midas)   
    print ("intraobjects accuracy midas")  
    print_accuracy(intra_midas)          
    print ("relative interobjects")
    print_accuracy(inter_relat)   
    print ("relative intraobjects")  
    print_accuracy(intra_relat)    
    print ("relative interobjects")
    print_accuracy(inter_relat_log)   
    print ("relative intraobjects")  
    print_accuracy(intra_relat_log)
                
                
    

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument("approach",
                        type=str,
                        help="which approach to use (natural2comics, depth_estimator, baseline)")
    
    parser.add_argument("model",
                        type=str,
                        help="which model to use (simple, depth_aware, add_text, add_text_ignoreloss, no_batchnorm_trick, batchnorm_trick)")
    
    args = parser.parse_args()

    evaluate(args.approach, agrs.model)
