import nibabel as nib
import torch
from torch.utils.data import Dataset
import os


class DataLoaderImg(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None, target_transform=None):
        self.image_dir = img_dir
        self.label_dir = annotations_dir
        self.labels = os.listdir(self.label_dir)
        self.images = os.listdir(self.image_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = nib.load(img_path)
        image = image.get_fdata()

        label_path = os.path.join(self.label_dir, self.labels[idx])
        label = nib.load(label_path)
        label = label.get_fdata()
        label = label

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        image = torch.from_numpy(image)
        image = image.unsqueeze_(0)
        label = torch.from_numpy(label)
        label = label.unsqueeze_(0)

        return image.float(), label.float()

