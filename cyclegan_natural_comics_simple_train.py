#!/usr/bin/env python
# coding: utf-8

def main():
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os


    unique_name = "cyclegan_natural_comics_simple"
    saving_folder = "models/trained/"+unique_name
    os.makedirs(saving_folder, exist_ok=True)


    import models.models

    from torch.utils.data import Dataset, DataLoader
    import glob
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms
    import numpy as np
    import random

    class ImageDataset_natural_comics(Dataset):

        def __init__(self,
                     unaligned = False):

            comics_file = "train.txt"
            list_comics = []
            with open(os.path.join("data/dcm_cropped", comics_file)) as file:
                content = file.read().split("\n")
                list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]    
            self.list_comics = list_comics
            self.list_natural = sorted(glob.glob("data/coco_val2017/*.*"))
            self.unaligned = unaligned

            self.t0 = Resize(
                        384,
                        384,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=1,
                        resize_method="lower_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    )
            #self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.t2 = PrepareForNet()
            self.tensorify = transforms.ToTensor()

        def __getitem__(self, index):
            index2 = random.randrange(0, len(self.list_comics)) if self.unaligned else index
            natural_name = self.list_natural[index % len(self.list_natural)]
            comics_name = self.list_comics[index2 % len(self.list_comics)]

            image_natural = cv2.imread(natural_name)
            image_comics = cv2.imread(comics_name)       


            if image_natural.ndim == 2:
                image_natural = cv2.cvtColor(image_natural, cv2.COLOR_GRAY2BGR)
            image_natural = cv2.cvtColor(image_natural, cv2.COLOR_BGR2RGB) / 255.0
            if image_comics.ndim == 2:
                image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0   


            #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
            #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
            #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


            #Resize to at least 384*834
            item_natural = self.t0({"image": image_natural})["image"]
            item_comics = self.t0({"image": image_comics})["image"]


            #NormalizeImage
            #item_natural = self.t1({"image": item_natural})["image"]
            #item_comics = self.t1({"image": item_comics})["image"] 

            #print("after resize :", natural_name, item_natural.shape[1], item_natural.shape[0])
            #if self.natural_depth: print("after resize :",depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
            #print("after resize :",comics_name, item_comics.shape[1], item_comics.shape[0])

            #Random crop to exactly 384*834     
            item_natural, _ = get_random_crop(item_natural, None, 384, 384)
            item_comics, _ = get_random_crop(item_comics, None, 384, 384)

            #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
            #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
            #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

            #PrepareForNet
            item_natural = self.t2({"image": item_natural})["image"]
            item_comics = self.t2({"image": item_comics})["image"]


            return {"natural": item_natural,
                    "comics": item_comics,
                   }

        def __len__(self):
            return max(len(self.list_comics), len(self.list_natural))

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


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")



    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = models.models.GeneratorResNet()
    natural2comics = models.models.GeneratorResNet()
    discrim_natural = models.models.Discriminator(input_shape = (3, 384, 384))
    discrim_comics = models.models.Discriminator(input_shape = (3, 384, 384))




    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()




    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = comics2natural.to(device)
    natural2comics = natural2comics.to(device)
    discrim_natural = discrim_natural.to(device)
    discrim_comics = discrim_comics.to(device)
    criterion_GAN = criterion_GAN.to(device)
    criterion_cycle = criterion_cycle.to(device)
    criterion_identity = criterion_identity.to(device)



    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_G = torch.optim.Adam(
        itertools.chain(comics2natural.parameters(), natural2comics.parameters()))
    optimizer_discrim_natural = torch.optim.Adam(discrim_natural.parameters())
    optimizer_discrim_comics = torch.optim.Adam(discrim_comics.parameters())




    dataloader = DataLoader(
        ImageDataset_natural_comics(unaligned = True),
        batch_size=4 if cuda else 1,
        shuffle=True,
        num_workers=8 if cuda else 0
    )




    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        comics2natural.load_state_dict(checkpoint['comics2natural_state_dict'])
        natural2comics.load_state_dict(checkpoint['natural2comics_state_dict'])
        discrim_natural.load_state_dict(checkpoint['discrim_natural_state_dict'])
        discrim_comics.load_state_dict(checkpoint['discrim_comics_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_discrim_natural.load_state_dict(checkpoint['optimizer_discrim_natural_state_dict'])
        optimizer_discrim_comics.load_state_dict(checkpoint['optimizer_discrim_comics_state_dict'])             
        epoch = checkpoint['epoch']
        print("LOADED CHECKPOINT EPOCH", epoch+1)
        epoch += 1
        print("RESUMING AT EPOCH", epoch+1)


    # Training
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

    while epoch+1 <= 100:

        for i, batch in enumerate(dataloader):

            # Set model input
            real_natural = batch["natural"].to(device)
            real_comics = batch["comics"].to(device)

            # Adversarial ground truths
            valid = torch.Tensor(np.ones((real_natural.size(0), *discrim_natural.output_shape))).to(device)
            fake = torch.Tensor(np.zeros((real_natural.size(0), *discrim_natural.output_shape))).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            natural2comics.train()
            comics2natural.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_natural = criterion_identity(comics2natural(real_natural), real_natural)
            loss_id_comics = criterion_identity(natural2comics(real_comics), real_comics)

            loss_identity = (loss_id_natural + loss_id_comics) / 2

            # GAN loss
            fake_comics = natural2comics(real_natural)
            loss_GAN_natural2comics = criterion_GAN(discrim_comics(fake_comics), valid)
            fake_natural = comics2natural(real_comics)
            loss_GAN_comics2natural = criterion_GAN(discrim_natural(fake_natural), valid)

            loss_GAN = (loss_GAN_natural2comics + loss_GAN_comics2natural) / 2



            # Cycle loss
            recov_natural = comics2natural(fake_comics)
            loss_cycle_natural = criterion_cycle(recov_natural, real_natural)
            recov_comics = natural2comics(fake_natural)
            loss_cycle_comics = criterion_cycle(recov_comics, real_comics)

            loss_cycle = (loss_cycle_natural + loss_cycle_comics) / 2

            # Total loss
            loss_G = loss_GAN + 10 * loss_cycle + 5 * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator natural
            # -----------------------
            optimizer_discrim_natural.zero_grad()

            # Real loss
            loss_real = criterion_GAN(discrim_natural(real_natural), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_natural(fake_natural.detach()), fake)
            # Total loss
            loss_discrim_natural = (loss_real + loss_fake) / 2

            loss_discrim_natural.backward()
            optimizer_discrim_natural.step()

            # -----------------------
            #  Train Discriminator comics
            # -----------------------
            optimizer_discrim_comics.zero_grad()

            # Real loss
            loss_real = criterion_GAN(discrim_comics(real_comics), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_comics(fake_comics.detach()), fake)
            # Total loss
            loss_discrim_comics = (loss_real + loss_fake) / 2

            loss_discrim_comics.backward()
            optimizer_discrim_comics.step()

            loss_D = (loss_discrim_natural + loss_discrim_comics) / 2



            # ----------
            print("epoch:", epoch+1, "i:", i+1,
                  "GANn2c:", loss_GAN_natural2comics.item(), "GANc2n:", loss_GAN_comics2natural.item(), "cycle:", loss_cycle.item(), "identity:", loss_identity.item(),
                  "Dn:", loss_discrim_natural.item(),"Dc:", loss_discrim_comics.item())

        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'comics2natural_state_dict': comics2natural.state_dict(),
                'natural2comics_state_dict': natural2comics.state_dict(),
                'discrim_natural_state_dict': discrim_natural.state_dict(),
                'discrim_comics_state_dict': discrim_comics.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_discrim_natural_state_dict': optimizer_discrim_natural.state_dict(),
                'optimizer_discrim_comics_state_dict': optimizer_discrim_comics.state_dict(),
                }
        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1





