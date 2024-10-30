"""
 This file was copied from https://github.com/tim-learn/SHOT-plus/code/uda/data_list.py and modified for this project needs.
 The license of the file is in: https://github.com/tim-learn/SHOT-plus/blob/master/LICENSE
"""

from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import os.path


###########
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

###########
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


###########
def make_dataset(image_list, labels):
    if labels is not None:
        images = image_list
    else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, img_root_dir=None):
        self.img_root_dir = img_root_dir
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders \n"))

        self.data = {}
        self.imgs = imgs
        self.transform = transform
        self.loader = rgb_loader

    def __getitem__(self, index):

        img_name, label = self.imgs[index]
        if self.img_root_dir is not None:
            img_name = os.path.join(self.img_root_dir, img_name)

        img = self.loader(img_name)
        img_tr = self.transform(img)

        return img_tr, label, img_name, index

    def __len__(self):
        return len(self.imgs)


class ImageList_twice(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, img_root_dir=None):
        self.img_root_dir = img_root_dir
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of\n"))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = rgb_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        if self.img_root_dir is not None:
            path = os.path.join(self.img_root_dir, path)
        img = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            if type(self.transform).__name__=='list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

