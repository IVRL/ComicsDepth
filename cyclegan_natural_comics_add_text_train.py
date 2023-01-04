#!/usr/bin/env python
# coding: utf-8

def main():
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os


    unique_name = "cyclegan_natural_comics_add_text"
    saving_folder = "models/trained"+unique_name
    os.makedirs(saving_folder, exist_ok=True)


    from torch.utils.data import Dataset, DataLoader
    import glob
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms
    import numpy as np
    import random

    class ImageDataset_natural_comicsWithBalloons(Dataset):

        def __init__(self,
                     unaligned = False):

            self.list_comics = []
            with open(os.path.join("data/dcm_cropped", "train.txt")) as file:
                content = file.read().split("\n")
                self.list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]
            self.list_natural = sorted(glob.glob("data/coco_val2017/*.*"))
            self.unaligned = unaligned

            self.t0 = Resize(
                        384,
                        384,
                        resize_target=True,
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
            balloons_name = comics_name.replace("dcm_cropped/images", "dcm_cropped/balloons427").replace(".jpg", "_originalsize.png")      


            image_natural = cv2.imread(natural_name)
            image_comics = cv2.imread(comics_name)   
            image_balloons = cv2.imread(balloons_name)    


            if image_natural.ndim == 2:
                image_natural = cv2.cvtColor(image_natural, cv2.COLOR_GRAY2BGR)
            image_natural = cv2.cvtColor(image_natural, cv2.COLOR_BGR2RGB) / 255.0
            if image_comics.ndim == 2:
                image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0 
            if image_balloons.ndim != 2:
                image_balloons = cv2.cvtColor(image_balloons, cv2.COLOR_BGR2GRAY)      
            image_balloons = image_balloons / 255.0


            #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
            #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
            #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


            #Resize to at least 384*834
            item_natural = self.t0({"image": image_natural})["image"]
            r = self.t0({"image": image_comics, "disparity":image_balloons})
            item_comics = r["image"]
            item_balloons = r["disparity"]


            #NormalizeImage
            #item_natural = self.t1({"image": item_natural})["image"]
            #item_comics = self.t1({"image": item_comics})["image"] 

            #print("after resize :", natural_name, item_natural.shape[1], item_natural.shape[0])
            #if self.natural_depth: print("after resize :",depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
            #print("after resize :",comics_name, item_comics.shape[1], item_comics.shape[0])

            #Random crop to exactly 384*834     
            item_natural, _ = get_random_crop(item_natural, None, 384, 384)
            item_comics, item_balloons = get_random_crop(item_comics, item_balloons, 384, 384)

            #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
            #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
            #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

            #PrepareForNet
            item_natural = self.t2({"image": item_natural})["image"]
            r = self.t2({"image": item_comics, "disparity":item_balloons})
            item_comics = r["image"]
            item_balloons = r["disparity"]


            return {"natural": item_natural,
                    "comics": item_comics,
                    "balloons": item_balloons,
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

                if "mask" in sample:
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


    #Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/models.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    #https://discuss.pytorch.org/t/how-do-i-print-output-of-each-layer-in-sequential/5773/3
    class PrintLayer(nn.Module):
        def __init__(self):
            super(PrintLayer, self).__init__()

        def forward(self, x):
            # Do your print / debug stuff here
            print(x.size())
            return x



    ##############################
    #           RESNET
    ##############################


    class ResidualBlock(nn.Module):
        def __init__(self, in_features):
            super(ResidualBlock, self).__init__()

            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                nn.InstanceNorm2d(in_features),
            )

        def forward(self, x):
            return x + self.block(x)


    class GeneratorResNet(nn.Module):
        def __init__(self, input_shape=(3, 256, 256), out_channels=3, num_residual_blocks=8):
            super(GeneratorResNet, self).__init__()

            channels = input_shape[0]

            # Initial convolution block
            out_features = 64
            model = [
                nn.ReflectionPad2d(3),
                #PrintLayer(),
                nn.Conv2d(channels, out_features, 7),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                #PrintLayer()
            ]
            in_features = out_features

            # Downsampling
            for _ in range(2):
                out_features *= 2
                model += [
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                    #PrintLayer()
                ]
                in_features = out_features

            # Residual blocks
            for _ in range(num_residual_blocks):
                model += [ResidualBlock(out_features)]

            # Upsampling
            for _ in range(2):
                out_features //= 2
                model += [
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                    #PrintLayer()
                ]
                in_features = out_features

            # Output 
            model += [nn.ReflectionPad2d(3),
                      nn.Conv2d(out_features, out_channels, 7),
                      nn.Sigmoid(), 
                      #PrintLayer()
                     ]

            self.model = nn.Sequential(*model)

        def forward(self, x):
            return self.model(x)

    ##############################
    #        Discriminator
    ##############################


    class Discriminator(nn.Module):
        def __init__(self, input_shape=(3, 256, 256)):
            super(Discriminator, self).__init__()

            channels, height, width = input_shape

            # Calculate output shape of image discriminator (PatchGAN)
            self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

            def discriminator_block(in_filters, out_filters, normalize=True):
                """Returns downsampling layers of each discriminator block"""
                layers = [
                    nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                    #PrintLayer()
                ]
                if normalize:
                    layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *discriminator_block(channels, 64, normalize=False),
                *discriminator_block(64, 128),
                *discriminator_block(128, 256),
                *discriminator_block(256, 512),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(512, 1, 4, padding=1)
            )

        def forward(self, img):
            return self.model(img)



    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")

    
    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = GeneratorResNet(input_shape=(3, 384, 384), out_channels=3, num_residual_blocks=8)
    generator_comicsB = GeneratorResNet(input_shape=(7, 384, 384), out_channels=3, num_residual_blocks=8)
    discrim_comicsB = Discriminator(input_shape = (3, 384, 384))



    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_GAN = torch.nn.MSELoss()
    criterion_balloons = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()


    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = natural2comicsNoB.to(device)
    generator_comicsB = generator_comicsB.to(device)
    discrim_comicsB = discrim_comicsB.to(device)
    criterion_GAN = criterion_GAN.to(device)
    criterion_balloons = criterion_balloons.to(device)
    criterion_identity = criterion_identity.to(device)



    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_G = torch.optim.Adam(generator_comicsB.parameters())
    optimizer_discrim_comicsB = torch.optim.Adam(discrim_comicsB.parameters())



    batch_size=16 if cuda else 1

    dataloader = DataLoader(
        ImageDataset_natural_comicsWithBalloons(unaligned = True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8 if cuda else 0
    )



    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_without_text", "99.pth")
    checkpoint = torch.load(checkpoint_file)
    natural2comicsNoB.load_state_dict(checkpoint['natural2comics_state_dict'])
    natural2comicsNoB = natural2comicsNoB.to(device)
    natural2comicsNoB.eval()



    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        generator_comicsB.load_state_dict(checkpoint['generator_comicsB_state_dict'])
        discrim_comicsB.load_state_dict(checkpoint['discrim_comicsB_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_discrim_comicsB.load_state_dict(checkpoint['optimizer_discrim_comicsB_state_dict'])           
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
            real_comicsB = batch["comics"].to(device)
            real_balloons = batch["balloons"].to(device).unsqueeze(1)

            natural2comicsNoB.eval()
            real_comicsNoB = natural2comicsNoB(real_natural).detach()


            # Adversarial ground truths
            valid = torch.Tensor(np.ones((real_natural.size(0), *discrim_comicsB.output_shape))).to(device)
            fake = torch.Tensor(np.zeros((real_natural.size(0), *discrim_comicsB.output_shape))).to(device)

            # ------------------
            #  Train Generator
            # ------------------
            generator_comicsB.train()
            optimizer_G.zero_grad()


            # Identity loss
            withItself = torch.cat((real_comicsB, real_balloons, real_comicsB), dim=1)
            loss_identity = criterion_identity(generator_comicsB(withItself), real_comicsB)

            # GAN loss
            inp = torch.cat((real_comicsB, real_balloons, real_comicsNoB), dim=1)
            fake_comicsB = generator_comicsB(inp)

            loss_GAN = criterion_GAN(discrim_comicsB(fake_comicsB), valid)

            # No change where no balloons loss
            loss_noB = criterion_balloons(real_comicsNoB*(1-real_balloons), fake_comicsB*(1-real_balloons))
            # No change where balloons loss
            loss_B = criterion_balloons(real_comicsB*real_balloons, fake_comicsB*real_balloons)

            # Total loss
            loss_G = loss_GAN + 3 * loss_noB + 3 * loss_B  + 5 * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator
            # -----------------------
            discrim_comicsB.train()
            optimizer_discrim_comicsB.zero_grad()

            # Real loss
            loss_real = criterion_GAN(discrim_comicsB(real_comicsB), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_comicsB(fake_comicsB.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_discrim_comicsB.step()



            # ----------

            print("epoch:", epoch+1, "i:", i+1,
                  "GAN:", loss_GAN.item(), "noB:", loss_noB.item(), "B:", loss_B.item(), "identity:", loss_identity.item(),
                  "D:", loss_D.item())

        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'generator_comicsB_state_dict': generator_comicsB.state_dict(),
                'discrim_comicsB_state_dict': discrim_comicsB.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_discrim_comicsB_state_dict': optimizer_discrim_comicsB.state_dict(),
                }

        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1



