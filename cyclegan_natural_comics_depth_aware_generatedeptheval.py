def main():

    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from PIL import Image
    import os


    unique_name = "apporoach1v1"

    import models.models


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")


    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")



    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    midas = midas.to(device)



    from torch.utils.data import Dataset, DataLoader
    import glob
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms
    import numpy as np
    import random

    class ImageDataset_comics(Dataset):

        def __init__(self):

            comics_file = "validation.txt"
            list_comics = []
            with open(os.path.join("data/dcm_cropped", comics_file)) as file:
                content = file.read().split("\n")
                list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]    
            self.list_comics = list_comics

            self.t0 = Resize(
                        384,
                        384,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    )
            #self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.t2 = PrepareForNet()
            self.tensorify = transforms.ToTensor()

        def __getitem__(self, index):
            comics_name = self.list_comics[index % len(self.list_comics)]

            image_comics = cv2.imread(comics_name)       


            if image_comics.ndim == 2:
                image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0   


            #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
            #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
            #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


            #Resize to at least 384*834
            x, y = image_comics.shape[1], image_comics.shape[0]
            item_comics = self.t0({"image": image_comics})["image"]


            #NormalizeImage
            #item_natural = self.t1({"image": item_natural})["image"]
            #item_comics = self.t1({"image": item_comics})["image"] 

            #print("after resize :", natural_name, item_natural.shape[1], item_natural.shape[0])
            #if self.natural_depth: print("after resize :",depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
            #print("after resize :",comics_name, item_comics.shape[1], item_comics.shape[0])

            #Random crop to exactly 384*834     
            #item_comics, _ = get_random_crop(item_comics, None, 384, 384)

            #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
            #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
            #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

            #PrepareForNet
            item_comics = self.t2({"image": item_comics})["image"]


            return {
                    "comics": item_comics,
                    "size": (x,y),
                    "name":comics_name
                   }

        def __len__(self):
            return len(self.list_comics)

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



    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = models.models.GeneratorResNet()
    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = comics2natural.to(device)


    #We want to evaluate epoch 100
    epoch = 100
    epoch -= 1

    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_depth_aware", str(epoch)+".pth")
    assert os.path.exists(checkpoint_file)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    comics2natural.load_state_dict(checkpoint['comics2natural_state_dict'])             
    epoch = checkpoint['epoch']
    print("LOADED CHECKPOINT EPOCH", epoch+1)

    comics2natural.eval()
    midas.eval()


    batch_size = 1

    dataloader = DataLoader(
        ImageDataset_comics(),
        batch_size=batch_size,
        #shuffle=True,
        num_workers= 8 if cuda else 0
    )


    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.Tensor(mean).view(3,1,1).to(device)
    std = torch.Tensor(std).view(3,1,1).to(device)

    for n, batch in enumerate(dataloader):
        #print(n)

        images = batch["comics"]

        images = images.to(device)

        with torch.no_grad():
            fake_natural = comics2natural(images)

            fake_natural -= mean
            fake_natural /= std
            prediction = midas(fake_natural)

        #print(images.data)
        #print(prediction.data)

        for i in range(batch["comics"].size()[0]):
            maxi = torch.max(prediction[i].view(-1))
            pred = prediction[i]/maxi
            pred = pred.unsqueeze(0).unsqueeze(0)

            # Resize to original resolution
            pred_resized = torch.nn.functional.interpolate(
                pred,
                size=(batch["size"][1][i], batch["size"][0][i]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            new_name = batch["name"][i].replace("dcm_cropped/images", "dcm_cropped/"+unique_name+"epoch"+str(epoch))
            print (new_name)
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            save_image(pred.squeeze(0).cpu(), new_name.replace(".jpg", ".png"))
            save_image(pred_resized.squeeze(0).cpu(), new_name.replace(".jpg", "_originalsize.png"))
            with open(new_name.replace(".jpg", ".txt"), "w+") as file:
                file.write(str(maxi.item()))



