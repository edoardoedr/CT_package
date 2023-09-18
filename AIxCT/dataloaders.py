import torch
import numpy as np
from torchvision import transforms
import os
import glob
from PIL import Image


class DataLoaderSegmentation_gray(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode, tiles = 224):
        super(DataLoaderSegmentation_gray, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'img','*.*'))
        self.label_files = []
        self.tiles = tiles
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            lab_filename = image_filename
            label_filename_with_ext = "label_"+lab_filename+".tif"
            self.label_files.append(os.path.join(folder_path, 'label', label_filename_with_ext))

        # Data augmentation and normalization for training
        # Just normalization for validation
        if "val" == mode :
            self.transforms = transforms.Compose([
                transforms.CenterCrop((self.tiles, self.tiles)),
                #transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    transforms.RandomCrop((self.tiles, self.tiles)),
                    #transforms.ToTensor(),
                ])

    def convert(self,img, target_type_min, target_type_max, target_type):
            imin = img.min()
            imax = img.max()

            a = (target_type_max - target_type_min) / (imax - imin)
            b = target_type_max - a * imax
            new_img = (a * img + b).astype(target_type)
            return new_img

    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]
            img = Image.open(img_path)
            label = Image.open(label_path)
            # Concatenate image and label, to apply same transformation on both
            img_to_tensor = transforms.ToTensor()
            image_tensor = img_to_tensor(img)
            label_tensor = img_to_tensor(label)
            image_and_label_tensor = torch.stack([image_tensor, label_tensor])

            # Apply Transforms
            image_and_label_tensor = self.transforms(image_and_label_tensor)

            # Extract image and label
            image = image_and_label_tensor[0, 0, :, :]
            label = image_and_label_tensor[1, 0, :, :]
            image = image.unsqueeze(-1)
            label = label.unsqueeze(0)
            label = label * 255

            #  Convert to int64 and remove second dimension
            label = label.long().squeeze()

            return image, label

    def __len__(self):
        return len(self.img_files)
    
class DataLoaderSegmentation_rgb(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode, tiles = 224):
        super(DataLoaderSegmentation_rgb, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'img','*.*'))
        self.label_files = []
        self.tiles = tiles
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            lab_filename = image_filename
            label_filename_with_ext = "label_"+lab_filename+".tif"
            self.label_files.append(os.path.join(folder_path, 'label', label_filename_with_ext))

        # Data augmentation and normalization for training
        # Just normalization for validation
        if "val" == mode :
            self.transforms = transforms.Compose([
                transforms.CenterCrop((self.tiles, self.tiles)),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    transforms.RandomCrop((self.tiles, self.tiles)),
                    transforms.ToTensor()
                ])

    def convert(self,img, target_type_min, target_type_max, target_type):
            imin = img.min()
            imax = img.max()

            a = (target_type_max - target_type_min) / (imax - imin)
            b = target_type_max - a * imax
            new_img = (a * img + b).astype(target_type)
            return new_img

    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]
            img = Image.open(img_path)
            label = Image.open(label_path)

            # Concatenate image and label, to apply same transformation on both
            img = img.convert('RGB')
            image_np = np.asarray(img)[:,:,:3].copy()
            label_np =  np.asarray(label).copy()

            new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
            image_and_label_np = np.zeros(new_shape, image_np.dtype)
            image_and_label_np[:, :, 0:3] = image_np
            image_and_label_np[:, :, 3] = label_np
            image_and_label = Image.fromarray(image_and_label_np)

            # Apply Transforms
            image_and_label = self.transforms(image_and_label)

            # Extract image and label
            image = image_and_label[0:3, :, :]
            label = image_and_label[3, :, :].unsqueeze(0)
            image = np.swapaxes(image, 0,2)
            image = np.swapaxes(image, 0,1)

            # Normalize back from [0, 1] to [0, 255]
            label = label * 255
            #  Convert to int64 and remove second dimension
            label = label.long().squeeze()

            return image, label


    def __len__(self):
        return len(self.img_files)