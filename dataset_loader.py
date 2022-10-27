import glob
import os
import torch
from PIL import Image
import skimage.io as sk
import cv2


class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, rootDir, sourceTransform):
        self.rootDir = rootDir
        self.data = glob.glob(os.path.join(self.rootDir+'/*'))
        self.sourceTransform = sourceTransform
        print(self.data)
        return 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.sourceTransform:
            image = self.sourceTransform(image=image)["image"]

        return image


if __name__ == '__main__':

    print("dataset loader")
