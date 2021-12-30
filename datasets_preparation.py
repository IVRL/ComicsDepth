#!/usr/bin/env python
# coding: utf-8

"""
datasets_preparation.py: Main utilities to prepare the datasets
"""


from utils.data_structure import *


def crop_frames() -> None:
    """
    Extracts one image for each frame from the DCM dataset.
    (In the DCM dataset, one image corresponds to one page.
    In order to train our models, we want to have one image for each frame.)
    """

    import os
    import cv2

    # Check the DCM structure is fine
    ASSERT_DCM()

    # Create new structure
    CREATE_DCM_CROPPED()

    # For the 3 subsets
    for subset in ["train", "validation", "test"]:

        # Get all images in the original subset
        dcm_filenames_list = DCM_GET_FILES_LIST(DCM_FILENAMES[subset])

        # Create the "cropped" subset file
        with open(DCM_CROPPED_FILENAMES[subset], "w+") as new_subset_file:

            # Iterate over all images in the original subset
            for filename in dcm_filenames_list:

                # Get the full-page image
                image_filename = DCM_IMAGE_PATH_FROM_NAME(filename)
                img = cv2.imread(image_filename)

                # Get the annotations about the page
                annot_filename = DCM_ANNOTATIONS_PATH_FROM_NAME(filename)
                annotations = DCM_READ_ANNOTATIONS(annot_filename)

                # Iterate over annotations
                counter = 1
                for annotation in annotations:
                    class_id, x1, y1, x2, y2 = annotation

                    # If frame annotation
                    if class_id == 8:
                        print(filename, counter)

                        # Crop frame image in full-page image
                        cropped = img[y1:y2, x1:x2]

                        # Save frame image
                        new_image_filename = DCM_CROPPED_IMAGE_PATH_FROM_NAME(
                            filename+str(counter))
                        cv2.imwrite(new_image_filename, cropped)

                        # Add the new frame image in the good subset
                        new_subset_file.write(filename+str(counter)+"\n")
                        counter += 1

    # Check the obtained sturcture is fine
    ASSERT_DCM_CROPPED()

    return


def generate_text_masks() -> None:
    """
    Extracts one image for each frame from the eBDtheque dataset.
    Then extracts the text masks from the annotations.
    Then randomly split in train/validation/test.
    """

    #######
    # Creates cropped images and text masks
    #######

    import xml.etree.ElementTree as ET
    import numpy as np
    import glob
    import sys
    import os
    import cv2

    file_list = glob.glob('data/eBDtheque_database_v3/GT/*.svg')

    for file in file_list:

        tree = ET.parse(file)
        root = tree.getroot()

        image = None
        panels = []
        balloons = []

        for child in root:
            if child.tag == "{http://www.w3.org/2000/svg}svg":
                if child.attrib["class"] == 'Page':
                    for child2 in child:
                        if child2.tag == "{http://www.w3.org/2000/svg}image":
                            assert image == None
                            image = {"filename": child2.attrib['{http://www.w3.org/1999/xlink}href'],
                                     'width': int(child2.attrib['width']),
                                     'height': int(child2.attrib['height']),
                                     }
                elif child.attrib["class"] == 'Panel':
                    for child2 in child:
                        points = child2.attrib['points'].split(" ")
                        points = [[int(x) for x in pt.split(",")]
                                  for pt in points]
                        assert points[0] == points[4]
                        panels.append((child2.attrib["id"], points))
                elif child.attrib["class"] == 'TextArea':
                    for child2 in child:
                        print(child2.tag, child2.attrib, "TODO TextArea")
                        raise NotImplementedError
                    pass
                elif child.attrib["class"] == 'Balloon':
                    for child2 in child:
                        points = child2.attrib['points'].split(" ")
                        points = np.array(
                            [[int(x) for x in pt.split(",")] for pt in points])
                        balloons.append(points)
                    pass
                elif child.attrib["class"] == 'Character':
                    pass
                elif child.attrib["class"] == 'Line':
                    for child2 in child:
                        points = child2.attrib['points'].split(" ")
                        points = np.array(
                            [[int(x) for x in pt.split(",")] for pt in points])
                        balloons.append(points)
                    pass
                elif child.attrib["class"] == 'linkSBSC':
                    pass
                else:
                    pass

        print("image", image)

        img = cv2.imread('data/eBDtheque_database_v3/'+image['filename'][3:])
        assert img.shape[0] == image['height']
        assert img.shape[1] == image['width']
        new_folder = "data/eBDtheque_cropped/"
        os.makedirs(new_folder[:-1], exist_ok=True)

        mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for balloon in balloons:
            cv2.fillPoly(mask, [balloon], (255, 255, 255))

        for panel in panels:
            points = panel[1]
            x1 = min(int(points[0][0]), int(points[3][0]),
                     int(points[1][0]), int(points[2][0]))
            x2 = max(int(points[0][0]), int(points[3][0]),
                     int(points[1][0]), int(points[2][0]))
            y1 = min(int(points[0][1]), int(points[1][1]),
                     int(points[2][1]), int(points[3][1]))
            y2 = max(int(points[0][1]), int(points[1][1]),
                     int(points[2][1]), int(points[3][1]))
            cropped = img[y1:y2, x1:x2]
            cropped_mask = mask[y1:y2, x1:x2]
            cv2.imwrite(new_folder+panel[0]+'.bmp', cropped)
            cv2.imwrite(new_folder+panel[0]+'_mask.bmp', cropped_mask)

    #######
    # Randomly split in train/validation/test
    #######

    import random

    path = "data/eBDtheque_cropped/"
    file_list = glob.glob(path+'*.bmp')
    file_list = [x[23:] for x in file_list if '_mask.bmp' not in x]
    print(len(file_list))

    train = []
    test = []
    validation = []
    for x in file_list:
        rd = random.random()
        if rd < 0.1:
            test.append(x)
        elif rd < 0.2:
            validation.append(x)
        else:
            train.append(x)

    print(len(train), len(train)/len(file_list))
    print(len(test), len(test)/len(file_list))
    print(len(validation), len(validation)/len(file_list))

    with open(path+"train.txt", "w") as file:
        file.writelines([x+"\n" for x in train])
    with open(path+"test.txt", "w") as file:
        file.writelines([x+"\n" for x in test])
    with open(path+"validation.txt", "w") as file:
        file.writelines([x+"\n" for x in validation])

    return


def generate_evaluation() -> None:
    """
    Opens a graphical interface to create the depth ordering for evaluation.
    """
    # Partially based on https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/
    # Partially based on https://www.python-course.eu/tkinter_canvas.php
    
    import os
    import tkinter
    import cv2
    from PIL import Image
    from PIL import ImageTk
    class DepthOrderingStatus:
        """
        Represents the current status of the process of creating the depth ordering.
        """
        def __init__(self):
            self.annotation = []
            self.inter_object_level = 1
            self.intra_object_level = 1

    subset = "validation"  # ="test"
    with open("data/dcm_cropped/"+subset+".txt") as subset_file:
        lines = subset_file.readlines()

    for line in lines:
        img_name = line[:-1]
        depth_filename = os.path.join(
            "data/dcm_cropped/depth", img_name+".txt")
        img_filename = os.path.join("data/dcm_cropped/images", img_name+".jpg")
        img_full_filename = os.path.join(
            "data/dcm_dataset.git/images", img_name[:-1]+".jpg")
        if not os.path.exists(img_full_filename):
            img_full_filename = os.path.join(
                "data/dcm_dataset.git/images", img_name[:-2]+".jpg")
        os.makedirs(os.path.dirname(depth_filename), exist_ok=True)
        if os.path.exists(depth_filename):
            print(depth_filename, "already exists")
        else:
            print(depth_filename, "does not exist, will create it")

            fen = tkinter.Tk()
            fen.title(depth_filename)

            img = cv2.imread(img_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            print(img_full_filename)
            img_full = cv2.imread(img_full_filename)
            img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
            (h, w) = img_full.shape[:2]
            img_full = cv2.resize(
                img_full, (int(w*img.height()/h), img.height()))
            img_full = Image.fromarray(img_full)
            img_full = ImageTk.PhotoImage(img_full)

            canvas = tkinter.Canvas(
                fen, height=img.height(), width=img.width())
            canvas2 = tkinter.Canvas(
                fen, height=img_full.height(), width=img_full.width())

            depthOrderingStatus = DepthOrderingStatus()
            colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
            colors += ["black"]*500

            def paint(event):
                canvas.delete("all")
                canvas.create_image(0, 0, anchor=tkinter.NW, image=img)
                for pt in depthOrderingStatus.annotation:
                    l1, l2, x, y = pt
                    canvas.create_line(
                        x-5, y, x+5, y, fill=colors[l1], width=2)
                    canvas.create_line(
                        x, y-5, x, y+5, fill=colors[l1], width=2)
                    canvas.create_text(
                        x, y, fill="darkblue", font="Times 20 bold", text=str(l1)+"."+str(l2))
                canvas.update()
                return

            def add_point(event):
                new = (depthOrderingStatus.inter_object_level,
                       depthOrderingStatus.intra_object_level, event.x, event.y)
                depthOrderingStatus.annotation.append(new)
                paint(event)
                print(depthOrderingStatus.annotation)
                print(depthOrderingStatus.inter_object_level,
                      depthOrderingStatus.intra_object_level)
                return

            def lclick(event):
                add_point(event)
                depthOrderingStatus.intra_object_level += 1
                return

            def rclick(event):
                depthOrderingStatus.intra_object_level -= 1
                add_point(event)
                depthOrderingStatus.intra_object_level += 1
                return

            def mclick(event):
                depthOrderingStatus.inter_object_level += 1
                intra_object_level = 1
                add_point(event)
                intra_object_level += 1
                return

            def cancel(event):
                print("cancel", depthOrderingStatus.annotation.pop())
                paint(event)
                return

            def left(event):
                depthOrderingStatus.inter_object_level -= 1
                print(depthOrderingStatus.inter_object_level,
                      depthOrderingStatus.intra_object_level)
                return

            def right(event):
                depthOrderingStatus.inter_object_level += 1
                print(depthOrderingStatus.inter_object_level,
                      depthOrderingStatus.intra_object_level)
                return

            def down(event):
                depthOrderingStatus.intra_object_level -= 1
                print(depthOrderingStatus.inter_object_level,
                      depthOrderingStatus.intra_object_level)
                return

            def up(event):
                depthOrderingStatus.intra_object_level += 1
                print(depthOrderingStatus.inter_object_level,
                      depthOrderingStatus.intra_object_level)
                return

            canvas.bind("<Button-1>", lclick)
            canvas.bind("<Button-2>", mclick)
            canvas.bind("<Button-3>", rclick)
            fen.bind("<BackSpace>", cancel)
            fen.bind("<Left>", left)
            fen.bind("<Right>", right)
            fen.bind("<Up>", up)
            fen.bind("<Down>", down)

            canvas.create_image(0, 0, anchor=tkinter.NW, image=img)
            canvas2.create_image(0, 0, anchor=tkinter.NW, image=img_full)

            canvas.pack(side=tkinter.LEFT)
            canvas2.pack(side=tkinter.RIGHT)
            fen.mainloop()

            print(depth_filename)
            print(depthOrderingStatus.annotation)
            #x = ""
            # while x != "Yes" and x != "No":
            #x = input("accept? Yes/No")
            if True:  # x == "Yes":
                with open(depth_filename, "w+") as depth_file:
                    depth_file.write("\n".join(
                        [" ".join([str(nb) for nb in pt]) for pt in depthOrderingStatus.annotation]))

    return


def coco17_depth() -> None:
    """
    Generates the \"ground-truth\" depth of natural images
    (In order to train our models, we want to have the depth of natural images,
    which we compute using MiDaS.)
    """    
    
    import glob
    import torch
    from PIL import Image
    from PIL import ImageTk
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    import os
    from torch.utils.data import DataLoader

    from utils.custom_dataset import CustomDataset

    torch.hub.set_dir(".cache/torch/hub")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.eval()

    batch_size = 1
    
    dataset = CustomDataset(dcm = [],
                            coco17 = True,
                            eBDtheque = False,
                            eBDtheque_cropped = False,
                            eBDtheque_cropped_mask = False,
                            natural_depth = False,
                            unaligned = False,
                            resize = (384, 384),
                            interpolation = Image.BICUBIC,
                            resize_mode = "inference",
                            max_len = 0
                           )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=1
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device is", device)
    midas = midas.to(device)

    for n, batch in enumerate(dataloader):
        print(n)

        images = batch["img_coco17"]

        images = images.to(device)

        with torch.no_grad():
            prediction = midas(images)

        for i in range(batch["img_coco17"].size()[0]):
            maxi = torch.max(prediction[i].view(-1))
            pred = prediction[i]/maxi
            pred = pred.unsqueeze(0).unsqueeze(0)

            # Resize to original resolution
            pred_resized = torch.nn.functional.interpolate(
                pred,
                size=(batch["size_coco17"][0][i], batch["size_coco17"][1][i]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            new_name = batch["name_coco17"][i].replace(
                "coco_val2017", "coco_val2017_depth")
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            save_image(pred.squeeze(0).cpu(), new_name.replace(".jpg", ".png"))
            save_image(pred_resized.squeeze(0).cpu(),
                       new_name.replace(".jpg", "_originalsize.png"))
            with open(new_name.replace(".jpg", ".txt"), "w+") as file:
                file.write(str(maxi.item()))
    return


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cf',
        "--crop_frames",
        action='store_true',
        help="generating cropped images from the DCM dataset"
    )
    parser.add_argument(
        '-gtm',
        "--generate_text_masks",
        action='store_true',
        help="generating comics text areas mask from the eBDtheque dataset and spliting the eBDtheque dataset"
    )
    parser.add_argument(
        '-ge',
        "--generate_evaluation",
        action='store_true',
        help="generating depth ordering for evaluation"
    )
    parser.add_argument(
        '-cd',
        "--coco17_depth",
        action='store_true',
        help="generating \"ground-truth\" depth of natural images"
    )
    args = parser.parse_args()

    if args.crop_frames:
        crop_frames()
    if args.generate_text_masks:
        generate_text_masks()
    if args.generate_evaluation:
        generate_evaluation()
    if args.coco17_depth:
        coco17_depth()
