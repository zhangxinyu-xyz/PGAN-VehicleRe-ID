# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
import random
import copy
from data.transforms.transforms import HorizontalFlip, RandomCrop

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

def read_proposal(proposal_path, img_path, img_size, isflip=False, prop_num=0, y_start=0, x_start=0, output_size=None):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_proposal = False
    img_name = osp.basename(img_path)
    w, h = img_size
    output_size = (output_size[0], output_size[1]) if output_size is not None else (h, w)
    folder = img_path.split('/')[-2]
    proposal_path = osp.join(proposal_path, folder, img_name.replace('jpg', 'txt'))
    if not osp.exists(proposal_path):
        return None
        #raise IOError("{} does not exist".format(proposal_path))
    
    proposal = []
    while not got_proposal:
        try:
            for line in open(proposal_path, 'r'):
                line = line[0:-1].split(' ')
                obj, score, xmin, ymin, xmax, ymax = \
                    line[0], float(line[1]), int(line[2]), int(line[3]), int(line[4]), int(line[5])
                if isflip:
                    temp = int(xmin)
                    xmin = w - int(xmax)
                    xmax = w - temp
                xmin, ymin, xmax, ymax = \
                    max(xmin, 0), max(ymin, 0), min(xmax, w-1), min(ymax, h-1)
                proposal.append([obj, score, xmin, ymin, xmax, ymax, (h, w), (y_start, x_start), output_size])
            got_proposal = True
        except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(proposal_path))
                pass

    proposal.sort(key=lambda x: x[1], reverse=True)
    return proposal

class ImageProposalDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, proposal_path=None, proposal_num=0,
                 istrain=False, padding=0):
        self.dataset = dataset
        self.transform = transform
        self.proposal_path = proposal_path
        self.proposal_num = proposal_num
        self.istrain = istrain
        self.padding = padding
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img_name = os.path.basename(img_path)
        folder = img_path.split('/')[-2]
        mask_path = osp.join(self.proposal_path, folder, img_name.replace('jpg', 'npy'))
        mask = np.load(mask_path)
        img = read_image(img_path)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            #mask.sort(key=lambda x: x[1], reverse=True)
            if self.proposal_num > 0:
                mask = mask[-self.proposal_num:]

        return img, mask, pid, camid, img_path
