import os
from skimage.io import imread
import pickle
import numpy as np
import torchvision
import torch

def pickle_images(root, dest):
    images = []
    for img_name in os.listdir(root):
        img = imread(os.path.join(root, img_name)).transpose(2, 0, 1)
        images.append(torchvision.transforms.Resize((64, 64))(torch.tensor(img)).numpy())

    images = np.array(images).astype(np.uint8)
    pickle.dump(images, dest)


if __name__ == '__main__':

    ROOT = 'dub2_data' # Location of folders with images
    IMAGE_FOLDERS = ['celeba_hq_256', 'avg_blurred_images', 'gauss_blurred_images'] # folder names

    for img_folder_name in IMAGE_FOLDERS:
        path = os.path.join(ROOT, img_folder_name)
        save_folder = os.path.join(ROOT, 'pickled')
        os.makedirs(save_folder, exist_ok=True)
        dest = open(os.path.join(save_folder, img_folder_name), 'wb')
        pickle_images(path, dest)
        dest.close()
        print("Done with", img_folder_name)

