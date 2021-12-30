#!/usr/bin/env python
# coding: utf-8

"""
comics2textareas.py: Code for the model about text area detection in comics images

"""


from utils.data_structure import *
from utils.custom_dataset import CustomDataset

import torch
import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader


#from https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py#L48
class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
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
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

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
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

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
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"].astype(bool)

        return sample

class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample

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


#https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image
def get_random_crop(image, image2, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = 0 if max_x==0 else np.random.randint(0, max_x)
    y = 0 if max_y==0 else np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    if image2 is not None :
        crop2 = image2[y: y + crop_height, x: x + crop_width]
    else:
        crop2 = None
    return crop, crop2



class ComicsForBalloons(Dataset):

    def __init__(self, file="train.txt", augment=False):

        with open("data/eBDtheque_cropped/"+file) as f:
            self.file_list = f.readlines()
        self.file_list = [x[:-1] for x in self.file_list]

        self.t0 = Resize(
                    384,
                    384,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=1,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
        self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.t2 = PrepareForNet()

        self.augment = augment
        if self.augment :
            random.seed(42)
            self.aug = A.Compose([  
                              A.ToGray(p=.1), 
                              A.RandomBrightnessContrast(p=.8),    
                              A.RandomGamma(p=.8),    
                              #A.CLAHE(p=.8),    
                              #A.JpegCompression(p=.5),   
                              A.HorizontalFlip(p=.5), 
                              #A.GridDistortion(p=.8), 
                              #A.ElasticTransform(p=.8) 
                          ], p=1)


    def __getitem__(self, index):

        file = "data/eBDtheque_cropped/" + self.file_list[index % len(self.file_list)]
        file_mask = "data/eBDtheque_cropped/" +self.file_list[index % len(self.file_list)].replace(".bmp", "_mask.bmp")

        image = cv2.imread(file)
        mask = cv2.imread(file_mask)

        if self.augment :
            augmented = self.aug(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.
        if mask.ndim != 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  
        mask = mask.astype(bool)



        #print("before resize :", image.shape[1], image.shape[0])
        #print("before resize :", mask.shape[1], mask.shape[0])
        #print("before resize :")
        #print (np.min(image[:, :, 0]), np.max(image[:, :, 0]))
        #print (np.min(image[:, :, 1]), np.max(image[:, :, 1]))
        #print (np.min(image[:, :, 2]), np.max(image[:, :, 2]))

        #resize
        if image.shape[0]<384 or image.shape[1]<384 :
            r = self.t0({"image": image, "mask":mask})
            image = r["image"]
            mask = r["mask"]

        #print("after resize :", image.shape[1], image.shape[0])
        #print("after resize :", mask.shape[1], mask.shape[0])
        #print("after resize :")
        #print (np.min(image[:, :, 0]), np.max(image[:, :, 0]))
        #print (np.min(image[:, :, 1]), np.max(image[:, :, 1]))
        #print (np.min(image[:, :, 2]), np.max(image[:, :, 2]))

        #NormalizeImage
        image = self.t1({"image": image})["image"]


        #print("after NormalizeImage :")
        #print (np.min(image[:, :, 0]), np.max(image[:, :, 0]))
        #print (np.min(image[:, :, 1]), np.max(image[:, :, 1]))
        #print (np.min(image[:, :, 2]), np.max(image[:, :, 2]))

        #Random crop to exactly 384*834     
        image, mask = get_random_crop(image, mask, 384, 384)

        #print("after crop :")
        #print (np.min(image[:, :, 0]), np.max(image[:, :, 0]))
        #print (np.min(image[:, :, 1]), np.max(image[:, :, 1]))
        #print (np.min(image[:, :, 2]), np.max(image[:, :, 2]))

        #PrepareForNet
        r = self.t2({"image": image, "mask": mask})
        image = r["image"]
        mask = r["mask"]

        #print("after prepare :")
        #print (np.min(image[0]), np.max(image[0]))
        #print (np.min(image[1]), np.max(image[1]))
        #print (np.min(image[2]), np.max(image[2]))


        return {"image": image, "mask": mask}

    def __len__(self):
        return len(self.file_list)

    
    
    
    

def train():
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
    
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    
    batch_size = 16 if cuda else 2

    dataloader = DataLoader(
        ComicsForBalloons(file="train.txt", augment = True),
        batch_size=batch_size,
        shuffle=True,
        num_workers= 16 if cuda else 0
    )
    
    batch_size = 16 if cuda else 2

    dataloader_val = DataLoader(
        ComicsForBalloons(file="validation.txt", augment = False),
        batch_size=batch_size,
        shuffle=True,
        num_workers= 16 if cuda else 0
    )
    
    
    unet = unet.train().to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.01)
    
    epoch = 0
    
    loss_train = []
    loss_valid = []
    iou_train = []
    iou_valid = []

    while epoch+1 <= 450:
        print("epoch", epoch+1, "training : ", end="")
        running_loss = 0
        running_iou = 0
        nb_samples = 0
        unet.train()

        for i, batch in enumerate(dataloader):

            # Set model input
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)#.long()


            # -------
            #  Train 
            # -------

            optimizer.zero_grad()

            output = unet(image).squeeze(1)
            #output = torch.cat((1-output, output), 1)

            # Loss
            #print(output.size())
            #print(mask.size())
            loss = criterion(output, mask)

            loss.backward()
            optimizer.step()


            print(i+1, end=" ")
            running_loss += loss.item()


            # IOU https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
            intersection = torch.logical_and(mask, output>0.5)
            union = torch.logical_or(mask, output>0.5)
            iou_score = (torch.sum(intersection).item()+1e-6) / (torch.sum(union).item()+1e-6)
            running_iou += iou_score * image.size()[0]

            nb_samples += image.size()[0]

            '''
            if True:
                plot_image = image[0].cpu()
                # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                plot_image[0] *= 0.229
                plot_image[0] += 0.485
                plot_image[1] *= 0.224
                plot_image[1] += 0.456
                plot_image[2] *= 0.225
                plot_image[2] += 0.406
                #print (torch.min(plot_image[0]), torch.max(plot_image[0]))
                #print (torch.min(plot_image[1]), torch.max(plot_image[1]))
                #print (torch.min(plot_image[2]), torch.max(plot_image[2]))
                m = torch.nn.Threshold(0., 0.) #resizing makes a few values outside 0,1
                plot_image = m(plot_image)
                m = torch.nn.Threshold(-1., -1.)
                plot_image = -m(-plot_image)

                plot_image = plot_image.permute([1,2,0])

                #print (torch.min(output.detach()[0]), torch.max(output.detach()[0]))
                #print (torch.min(mask[0]), torch.max(mask[0]))

                imagesplot = [plot_image, output.cpu().detach()[0], mask[0].cpu()]
                plot(imagesplot, figsize=(16.,10.), nrows_ncols=(1, 3))
            '''

        running_loss /= nb_samples
        running_iou /= nb_samples 
        print('')
        print("epoch:", epoch+1, "training loss over epoch", running_loss, "iou (at 0.5)", running_iou)
        loss_train.append(running_loss)
        iou_train.append(running_iou)

        print("epoch:", epoch+1, "evaluating : ", end="")
        running_loss = 0
        running_iou = 0
        nb_samples = 0
        unet.eval()

        for i, batch in enumerate(dataloader_val):
            with torch.no_grad():
                # Set model input
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)#    .long()    
                # -------
                output = unet(image).squeeze(1)
                #output = torch.cat((1-output, output), 1)
                # Loss
                loss = criterion(output, mask)
                print(i+1, end=" ")
                running_loss += loss.item()
                # IOU https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
                intersection = torch.logical_and(mask, output>0.5)
                union = torch.logical_or(mask, output>0.5)
                iou_score = (torch.sum(intersection).item()+1e-6) / (torch.sum(union).item()+1e-6)
                running_iou += iou_score  * image.size()[0]

                nb_samples += image.size()[0]
        running_loss /= nb_samples
        running_iou /= nb_samples
        print('')
        print("epoch", epoch+1, "validation loss", running_loss, "iou (at 0.5)", running_iou)
        loss_valid.append(running_loss)
        iou_valid.append(running_iou)

        to_save = {
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': loss_train,
                'validation_loss_history': loss_valid,
                'train_iou_history': iou_train,
                'validation_iou_history': iou_valid,
                }
        saving_folder = "models/trained/comics2textareas/"
        os.makedirs(saving_folder, exist_ok=True)
        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))

        epoch += 1

def select_epoch():
        
    epoch = 449
    saving_folder = "models/trained/comics2textareas/"
    checkpoint_file = os.path.join(saving_folder, str(epoch)+".pth")
    print(checkpoint_file)

    assert os.path.exists(checkpoint_file)

    print("FOUND CHECKPOINT")
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    loss_train = checkpoint['train_loss_history']
    loss_valid = checkpoint['validation_loss_history']
    iou_train = checkpoint['train_iou_history']
    iou_valid = checkpoint['validation_iou_history']
    print("LOADED CHECKPOINT EPOCH", epoch+1)
    epoch += 1
    
    smooth = 10
    ln = len(iou_valid)
    x = range(smooth, ln-smooth)

    epoch = max(x, key = lambda i: sum([iou_train[i+j] for j in range(-smooth, smooth+1)]))
    print(epoch+1, iou_valid[epoch])

    
    
    
def optimize_threshold():
    
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
    epoch = 427
    
    saving_folder = "models/trained/comics2textareas/"
    checkpoint_file = os.path.join(saving_folder, str(epoch)+".pth")
    print(checkpoint_file)

    assert os.path.exists(checkpoint_file)

    print("FOUND CHECKPOINT")
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    print("LOADED CHECKPOINT EPOCH", epoch+1)
    
    
    batch_size = 16 if cuda else 2

    dataloader_val = DataLoader(
        ComicsForBalloons(file="validation.txt", augment = False),
        batch_size=batch_size,
        shuffle=True,
        num_workers= 16 if cuda else 0
    )
    
    
    t = [0.01 * i for i in range(101)]
    ious = []
    for threshold in t:
        print("threshold:", threshold, "evaluating : ", end="")
        running_iou = 0
        nb_samples = 0
        unet.eval()
        for i, batch in enumerate(dataloader_val):
            with torch.no_grad():
                # Set model input
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)

                output = unet(image).squeeze(1)

                # IOU https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
                intersection = torch.logical_and(mask, output>threshold)
                union = torch.logical_or(mask, output>threshold)
                iou_score = (torch.sum(intersection).item()+1e-6) / (torch.sum(union).item()+1e-6)
                running_iou += iou_score * image.size()[0]

                nb_samples += image.size()[0]

        running_iou /= nb_samples
        print("iou", running_iou)
        ious.append(running_iou)
    
    smooth = 15
    ln = 100
    t = range(smooth, ln-smooth)

    best = max(t, key = lambda i: sum([ious[i+j] for j in range(-smooth, smooth+1)]))
    print(best, best/100, ious[best])
    threshold =  best/100
    
    
def visualize(dataset):
    
    import glob
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from torch.utils.data import Dataset, DataLoader
    import os    
    
    import cv2
    import numpy as np

    class ComicsDatasetV2(Dataset):

        def __init__(self, train=True, val=True, test=True):
            self.list_comics = []
            if train: 
                with open(os.path.join("data/dcm_cropped", "train.txt")) as file:
                    content = file.read().split("\n")
                    self.list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]
            if val: 
                with open(os.path.join("data/dcm_cropped", "validation.txt")) as file:
                    content = file.read().split("\n")
                    self.list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]
            if test: 
                with open(os.path.join("data/dcm_cropped", "test.txt")) as file:
                    content = file.read().split("\n")
                    self.list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]

            self.transform = transforms.Compose(
                [
                    Resize(
                        384,
                        384,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ]
            )#from https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py#L48

        def __getitem__(self, index):
            #https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/utils.py#L98
            img = cv2.imread(self.list_comics[index % len(self.list_comics)])
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

            size = image.shape
            item = self.transform({"image": image})["image"]

            return {"img":item,
                    "name":self.list_comics[index % len(self.list_comics)],
                    "size":size
                   } 

        def __len__(self):
            return len(self.list_comics)  

    #from https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py#L48
    class Resize(object):
        """Resize sample to given size (width, height).
        """

        def __init__(
            self,
            width,
            height,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=1,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_AREA,
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
                y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

            if y < min_val:
                y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

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
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

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
                        sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                    )

                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["mask"] = sample["mask"].astype(bool)

            return sample

    class NormalizeImage(object):
        """Normlize image by given mean and std.
        """

        def __init__(self, mean, std):
            self.__mean = mean
            self.__std = std

        def __call__(self, sample):
            sample["image"] = (sample["image"] - self.__mean) / self.__std

            return sample

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


    #https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image
    def get_random_crop(image, image2, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height
        x = 0 if max_x==0 else np.random.randint(0, max_x)
        y = 0 if max_y==0 else np.random.randint(0, max_y)
        crop = image[y: y + crop_height, x: x + crop_width]
        if image2 is not None :
            crop2 = image2[y: y + crop_height, x: x + crop_width]
        else:
            crop2 = None
        return crop, crop2
    
    torch.hub.set_dir(".cache/torch/hub")
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=False)
    unet.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_file = "models/trained/comics2textareas/427.pth"
    checkpoint = torch.load(checkpoint_file, map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    unet.eval()
    
    batch_size = 1

    dataloader = DataLoader(
        ComicsDatasetV2(),
        batch_size=batch_size,
        num_workers= 0#1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device is", device)
    unet = unet.to(device)

    for n, batch in enumerate(dataloader):
        print(n)

        images = batch["img"]

        images = images.to(device)

        with torch.no_grad():
            prediction = unet(images)

        for i in range(batch["img"].size()[0]):
            pred = prediction[i]
            pred = pred.unsqueeze(0)

            # Resize to original resolution
            pred_resized = torch.nn.functional.interpolate(
                pred,
                size=(batch["size"][0][i], batch["size"][1][i]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            new_name = batch["name"][i].replace("dcm_cropped/images", "dcm_cropped/balloons427")
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            save_image(pred.squeeze(0).cpu(), new_name.replace(".jpg", ".png"))
            save_image(pred_resized.squeeze(0).cpu(), new_name.replace(".jpg", "_originalsize.png"))
    
    
def generate_dcm_without_text_areas():
    
    import glob
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from torch.utils.data import Dataset, DataLoader
    import os
    
    import cv2
    import numpy as np



    #from https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py#L48
    class Resize(object):
        """Resize sample to given size (width, height).
        """

        def __init__(
            self,
            width,
            height,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=1,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_AREA,
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
                y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

            if y < min_val:
                y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

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
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

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
                        sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                    )

                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["mask"] = sample["mask"].astype(bool)

            return sample

    class NormalizeImage(object):
        """Normlize image by given mean and std.
        """

        def __init__(self, mean, std):
            self.__mean = mean
            self.__std = std

        def __call__(self, sample):
            sample["image"] = (sample["image"] - self.__mean) / self.__std

            return sample

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


    #https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image
    def get_random_crop(image, image2, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height
        x = 0 if max_x==0 else np.random.randint(0, max_x)
        y = 0 if max_y==0 else np.random.randint(0, max_y)
        crop = image[y: y + crop_height, x: x + crop_width]
        if image2 is not None :
            crop2 = image2[y: y + crop_height, x: x + crop_width]
        else:
            crop2 = None
        return crop, crop2
    
    comics_file = "train.txt"
    list_comics = []
    with open(os.path.join("data/dcm_cropped", comics_file)) as file:
        content = file.read().split("\n")
        list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]       
    t0 = Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=1,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            )
    tensorify = transforms.ToTensor()

    os.makedirs("data/dcm_cropped_noballoons", exist_ok=True)

    for index in range(len(list_comics)):
        print(index, end = " ")
        comics_name = list_comics[index % len(list_comics)]
        image_comics = cv2.imread(comics_name)       
        balloons_name = comics_name.replace("dcm_cropped/images", "dcm_cropped/balloons427").replace(".jpg", "_originalsize.png")        
        image_balloons = cv2.imread(balloons_name)
        if image_comics.ndim == 2:
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
        image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0
        if image_balloons.ndim != 2:
            image_balloons = cv2.cvtColor(image_balloons, cv2.COLOR_BGR2GRAY)        


        #Resize to at least 384*834
        item_comics = t0({"image": image_comics})["image"]
        item_balloons = t0({"image": image_balloons})["image"]

        maxi = 384
        n = maxi
        done = False
        while n>maxi/4 and not done:
            test = 0
            print("("+str(n)+")", end = " ")
            while test<20 and not done:
                #Random crop to exactly n*n     
                item_comics_cropped, item_balloons_cropped = get_random_crop(item_comics, item_balloons, n, n)        
                done = item_balloons_cropped.max() < 0.03*255
                test += 1
            n=n-1
        print(done)
        if done:
            save_image(tensorify(item_comics_cropped), "data/dcm_cropped_noballoons/"+str(index)+".png")




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",        
        "--train",
        action='store_true',
        help="training the Unet"
    )
    parser.add_argument(
        "-se",       
        "--select_epoch",
        action='store_true',
        help="selecting epoch with highest validation IoU at threshold 0.5"
    )
    parser.add_argument(
        "-ot",
        "--optimize_threshold",
        action='store_true',
        help="optimizing the threshold"
    )
    parser.add_argument(
        "-vev",
        "--visualize_ebdtheque_val",
        action='store_true',
        help="visualizing the results on the validation set of the eBDtheque dataset"
    )
    parser.add_argument(
        "-vd",
        "--visualize_dcm",
        action='store_true',
        help="visualizing the results on the dcm dataset"
    )
    parser.add_argument(
        "-gdwta",
        "--generate_dcm_without_text_areas",
        action='store_true',
        help="using the trained model to generate a dataset of comics without text areas"
    )
    args = parser.parse_args()

    if args.train:
        train()
    if args.select_epoch:
        select_epoch()
    if args.optimize_threshold:
        optimize_threshold()
    if args.visualize_ebdtheque_val:
        visualize("eBDtheque")
    if args.visualize_dcm:
        visualize("dcm")
    if args.generate_dcm_without_text_areas:
        generate_dcm_without_text_areas()
