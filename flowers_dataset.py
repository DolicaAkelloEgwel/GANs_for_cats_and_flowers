import os
import random
import tarfile
from urllib import request

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import preprocess_img


class Flowers64Dataset(Dataset):
    """Dataset object for 64x64 pixel flower images."""

    def __init__(self, img_paths, mirror=True):
        self.size = 64
        self.mirror = mirror

        self.images = [cv2.imread(image) for image in img_paths]
        self._preprocess_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # mirror img with a 50% chance
        if self.mirror:
            if random.random() > 0.5:
                img = img[:, ::-1, :]

        return torch.tensor(img.astype(np.float32))
    
    def _preprocess_images(self):
        for i in range(len(self.images)):
            # center crop
            h, w = self.images[i].shape[:2]
            if h != w:
                min_side = min(h, w)
                top, bot = (h - min_side) // 2, h - (h - min_side) // 2
                left, right = (w - min_side) // 2, w - (w - min_side) // 2
                self.images[i] = self.images[i][top:bot, left:right, :]

            # resize
            self.images[i] = cv2.resize(self.images[i], (self.size, self.size))

            # normalize
            self.images[i] = preprocess_img(self.images[i])

        print(self.images[0].shape)


    @classmethod
    def create_from_scratch(cls, data_path):
        """Download and extract data, and create dataset object."""
        # download images
        flower_img_paths = cls.prepare_flowers_data(data_path)
        # define dataset
        dataset = cls(img_paths=flower_img_paths)
        return dataset

    @staticmethod
    def prepare_flowers_data(data_path="data"):
        """_summary_

        Args:
            data_path (str, optional): _description_. Defaults to "data".

        Returns:
            _type_: _description_
        """
        flowers_data_path = os.path.join(data_path, "flower_data")
        if not os.path.exists(flowers_data_path) or not os.listdir(flowers_data_path):
            os.makedirs(flowers_data_path, exist_ok=True)
            print("Downloading flower dataset...")
            local_filename, _ = request.urlretrieve(
                "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
            )
            flowers_dataset = tarfile.open(local_filename)
            flowers_dataset.extractall(flowers_data_path)
        else:
            print("Dataset already prepared in {}".format(flowers_data_path))

        img_dir = os.path.join(flowers_data_path, "jpg")
        img_paths = [
            os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")
        ]
        img_paths.sort()
        return img_paths
