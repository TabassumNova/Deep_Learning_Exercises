from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
from skimage import transform
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        #self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])
        self._transform = None
        self.root_dir = "C:/NOVA/MSC@FAU/Deep Learning/Exercise/exercise4_material/exercise4_material/src_to_implement/images"

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data.iloc[index, 0])
        image = imread(img_name)
        image_shape = image.shape[0]
        image = gray2rgb(image).reshape(3,image_shape, image_shape)
        #print(image.shape)
        # image = image.reshape((-1, 3))
        # print(image.shape)
        # image = np.transpose(image, (0, 1, 0)) # 0 1 2 -> 2 1 0
        # print(image.shape)
        #print(self.data.iloc[:5, 1:])
        label = self.data.iloc[index, 1:]

        label = np.array([label])
        label = label.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        sample = (image, label)

        if self._transform:
            sample = self._transform(sample)

        return sample


