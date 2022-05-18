import cv2
import os
from os import listdir
from os.path import isfile, join
import random
import matplotlib.pyplot as plt


class Dataset():
    def __init__(self, folder):
        self.path = folder
        self.image_paths = [f for f in sorted(listdir(
            self.path)) if isfile(join(self.path, f))]
        self.subset = Subset(self)

    def __str__(self):
        images = []
        for i in range(len(self.image_paths)):
            images.append(self.image_paths[i])
            if i == 5:
                break
        return str(images)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx, read=False):
        items = self.image_paths[idx]
        if read:
            return list(map(self.read_image, items))
        return items

    def read_image(self, image_path, scale=1.):
        if not self.is_image(image_path):
            image_path = image_path + ".jpg"

        if not (isfile(image_path)):
            image_path = os.path.abspath(join(self.path, image_path))

        if not (isfile(image_path)):
            raise FileNotFoundError(image_path)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    def show_image(self, image, gray=False, **kwargs):
        if isinstance(image, str):
            image = self.read_image(image)
        if not gray:
            plt.imshow(image, aspect="equal", **kwargs)
        else:
            plt.imshow(image, aspect="equal", cmap="gray", **kwargs)

    def is_image(self, path):
        allowed_extensions = [
            ".jpeg", ".jpg", ".jp2",
            ".png",
            ".bmp",
            ".tif", ".tiff",
            "pbm", ".pgm", "ppm"]
        return os.path.splitext(path)[-1] in allowed_extensions


class Subset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        subset = self.dataset
        subset.image_paths = subset.image_paths[idx]
        return subset
