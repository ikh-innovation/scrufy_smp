import cv2
import numpy as np

import torch 
from torch.utils.data import Dataset


class BluebotDataset(Dataset):
    """
    Class to build the custom dataset.
    Inherits from the Dataset class of pytorch.
    Performs necessary preprocessing and transform to prepare images for model
    
    """
    def __init__(self, inputs: list, targets: list, transform=None ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        # Select the sample
        input_id = self.inputs[idx]
        target_id = self.targets[idx]

        # Load input and target
        img, mask = cv2.imread(input_id), cv2.imread(target_id, cv2.IMREAD_GRAYSCALE)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.expand_dims(mask, axis=-1)
        img = img.transpose(2, 1, 0)
        mask = mask.transpose(2, 1, 0)
        # Preprocessing
        if self.transform:
            img = self.transform(image=img)
        img = img['image']
        img, mask = torch.from_numpy(img.copy()).type(self.inputs_dtype), torch.from_numpy(mask.copy()).type(self.targets_dtype)

        return img, mask