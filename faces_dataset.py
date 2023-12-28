import os
from typing import Any, Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from skimage.io import imread
from torchvision.datasets import VisionDataset
import pickle


class FacesDataset(VisionDataset):
    def __init__(
            self,
            file_path: str,
            transform: Optional[Callable] = None
    ) -> None:
        super().__init__(file_path, transform=transform)
        self.transform = transform

        file = open(file_path, 'rb')
        self.data = pickle.load(file).transpose(0, 2, 3, 1)
        file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.data[index]

        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img
