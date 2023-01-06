def main(lr):
    
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os

    unique_name = "depth_add_text_ignoretext_"+str(lr)
    saving_folder = "models/trained/"+unique_name
    os.makedirs(saving_folder, exist_ok=True)


    from torch.utils.data import Dataset, DataLoader
    import glob
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms
    import numpy as np
    import random

    class ImageDataset_naturalWithDepth_comicsWithBalloons(Dataset):

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

            depth_name = natural_name.replace("coco_val2017", "coco_val2017_depth").replace(".jpg", ".png")
            with open(depth_name.replace(".png", ".txt")) as file:
                scale = float(file.read())        

            image_natural = cv2.imread(natural_name)
            image_comics = cv2.imread(comics_name)   
            image_balloons = cv2.imread(balloons_name)    
            image_natural_depth = cv2.imread(depth_name)


            if image_natural.ndim == 2:
                image_natural = cv2.cvtColor(image_natural, cv2.COLOR_GRAY2BGR)
            image_natural = cv2.cvtColor(image_natural, cv2.COLOR_BGR2RGB) / 255.0
            if image_comics.ndim == 2:
                image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0 
            if image_balloons.ndim != 2:
                image_balloons = cv2.cvtColor(image_balloons, cv2.COLOR_BGR2GRAY)      
            image_balloons = image_balloons / 255.0
            if image_natural_depth.ndim != 2:
                image_natural_depth = cv2.cvtColor(image_natural_depth, cv2.COLOR_BGR2GRAY)      
            image_natural_depth = image_natural_depth / 255.0


            #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
            #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
            #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


            #Resize to at least 384*834
            r = self.t0({"image": image_natural, "disparity":image_natural_depth})
            item_natural = r["image"]
            item_natural_depth = r["disparity"]

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
            item_natural, item_natural_depth = get_random_crop(item_natural, item_natural_depth, 384, 384)
            item_comics, item_balloons = get_random_crop(item_comics, item_balloons, 384, 384)

            #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
            #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
            #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

            #PrepareForNet
            r = self.t2({"image": item_natural, "disparity":item_natural_depth})
            item_natural = r["image"]
            item_natural_depth = r["disparity"]
            r = self.t2({"image": item_comics, "disparity":item_balloons})
            item_comics = r["image"]
            item_balloons = r["disparity"]

            item_natural_depth *= scale


            return {"natural": item_natural,
                    "natural_depth": item_natural_depth,
                    "scale": scale,
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


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")

    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = GeneratorResNet(input_shape=(3, 384, 384), out_channels=3, num_residual_blocks=8)
    comicsNoB2comics = GeneratorResNet(input_shape=(7, 384, 384), out_channels=3, num_residual_blocks=8)
    comics2depth = torch.hub.load("intel-isl/MiDaS", "MiDaS")


    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_depth = torch.nn.L1Loss()



    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = natural2comicsNoB.to(device)
    comicsNoB2comics = comicsNoB2comics.to(device)
    comics2depth = comics2depth.to(device)
    criterion_depth = criterion_depth.to(device)


    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_comics2depth = torch.optim.Adam(comics2depth.parameters(), lr=lr)


    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_without_text", "99.pth")
    checkpoint = torch.load(checkpoint_file)
    natural2comicsNoB.load_state_dict(checkpoint['natural2comics_state_dict'])
    natural2comicsNoB = natural2comicsNoB.to(device)
    natural2comicsNoB.eval()



    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_add_text", "99.pth")
    checkpoint = torch.load(checkpoint_file)
    comicsNoB2comics.load_state_dict(checkpoint['generator_comicsB_state_dict'])
    comicsNoB2comics = comicsNoB2comics.to(device)
    comicsNoB2comics.eval()


    comics2depth.train()



    batch_size=16 if cuda else 1

    dataloader = DataLoader(
        ImageDataset_naturalWithDepth_comicsWithBalloons(unaligned = True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=16 if cuda else 0
    )



    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        comics2depth.load_state_dict(checkpoint['comics2depth_state_dict'])
        optimizer_comics2depth.load_state_dict(checkpoint['optimizer_comics2depth_state_dict'])         
        epoch = checkpoint['epoch']
        print("LOADED CHECKPOINT EPOCH", epoch+1)
        epoch += 1
        print("RESUMING AT EPOCH", epoch+1)


    natural2comicsNoB.eval()
    comicsNoB2comics.eval()
    comics2depth.train()

    # Training
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.Tensor(mean).view(3,1,1).to(device)
    std = torch.Tensor(std).view(3,1,1).to(device)


    while epoch+1 <= 100:

        for i, batch in enumerate(dataloader):

            real_natural = batch["natural"].to(device)
            real_natural_depth = batch["natural_depth"].to(device)
            real_comicsB = batch["comics"].to(device)
            real_balloons = batch["balloons"].to(device).unsqueeze(1)

            with torch.no_grad():
                comicsNoB = natural2comicsNoB(real_natural).detach()
                inp = torch.cat((real_comicsB, real_balloons, comicsNoB), dim=1)
                fake_comicsB = comicsNoB2comics(inp).detach()
                fake_comicsB -= mean
                fake_comicsB /= std


            optimizer_comics2depth.zero_grad()
            pred_depth = comics2depth(fake_comicsB.detach())
            loss = criterion_depth(pred_depth*(1-real_balloons), real_natural_depth*(1-real_balloons))

            loss.backward()
            optimizer_comics2depth.step()


            # ----------

            print("epoch:", epoch+1, "i:", i+1,
                  "loss:", loss.item())

        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'comics2depth_state_dict': comics2depth.state_dict(),
                'optimizer_comics2depth_state_dict': optimizer_comics2depth.state_dict(),
                }

        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1

