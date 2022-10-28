import glob
import os
import torch
from PIL import Image
import skimage.io as sk
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
image_size = 28


class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.data = glob.glob(os.path.join(self.rootDir+'/*'))
        self.sourceTransform = A.Compose(
            [
                A.SmallestMaxSize(max_size=image_size),
                A.CenterCrop(height=image_size, width=image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        return 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        name = self.data[idx].split("/")[-1]

        if self.sourceTransform:
            image = self.sourceTransform(image=image)["image"]

        return image, name


if __name__ == '__main__':

    print("dataset loader")
