import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation

class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


class VOCSeg(Dataset):
    CLASSES = ["background","aeroplane","bicycle","bird","boad","bottle",
                "bus","car","cat","chair","cow","dining table","dog","horse",
                "motor bike","person","potted plant","sheep","sofa","train",
                "tv/monitor","void"]

    def __init__(self, root="/workspace/datasets", image_set="val", image_size=448, crop=False):
        if crop:
            transform = torchvision.transforms.Compose([torchvision.transforms.Resize((672, 672)), torchvision.transforms.RandomCrop(crop), torchvision.transforms.Resize((image_size, image_size)), torchvision.transforms.ToTensor()])
            target_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((672, 672), Image.NEAREST), torchvision.transforms.RandomCrop(crop), torchvision.transforms.Resize((image_size, image_size), Image.NEAREST), ToTargetTensor()])
        else:
            transform = torchvision.transforms.Compose([torchvision.transforms.Resize((image_size, image_size)), torchvision.transforms.ToTensor()])
            target_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((image_size, image_size), Image.NEAREST), ToTargetTensor()])
        self.dataset = VOCSegmentation(root=root, year='2012', image_set=image_set, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        mask[mask==255] = 21
        mask -= 1
        mask[mask<0] = 20
        return image, mask[0]

def load_voc_dataloader(is_val=True, batch_size=8, image_size=448, crop=False):
    if is_val:
        dataset = VOCSeg(root="/workspace/datasets", image_set='val', image_size=image_size, crop=crop)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    else:
        dataset = VOCSeg(root="/workspace/datasets", image_set='train', image_size=image_size, crop=crop)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

